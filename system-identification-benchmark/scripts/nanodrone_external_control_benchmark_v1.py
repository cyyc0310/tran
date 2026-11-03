#!/usr/bin/env python3
"""Frozen-budget external system-ID evaluation on the public Nano-Drone data.

The raw CSVs and author repository are external, pinned inputs; this script
verifies each CSV against the tracked Git-LFS pointer before scoring anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import solve


ROOT = Path(__file__).resolve().parents[1]
DT = 0.01
HORIZON = 50
OBSERVATION_STEP = 50
TARGET_BUDGETS_SECONDS = (0, 1, 3, 5)
GAMMA_GRID = np.linspace(0.0, 2.0, 41)
ARX_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
SINDY_THRESHOLD_GRID = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def clean_json(value: Any) -> Any:
    """Convert NumPy scalars and non-finite values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def best_finite_index(scores: list[float], label: str) -> int:
    valid = [index for index, score in enumerate(scores) if np.isfinite(score)]
    if not valid:
        raise RuntimeError(f"All {label} candidates failed on source validation")
    return min(valid, key=scores.__getitem__)


def verify_tracked_file_at_commit(author_root: Path, relative_path: str, commit: str) -> Path:
    """Reject locally edited model/scaler files even when HEAD is pinned."""
    path = author_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned author file: {path}")
    committed = subprocess.run(
        ["git", "-C", str(author_root), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout
    if sha256(path) != sha256_bytes(committed):
        raise ValueError(f"Working file differs from pinned author commit: {relative_path}")
    return path


def rotvec_from_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    """PyTorch3D-compatible SO(3) logarithm for unit (x,y,z,w) quaternions."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-15)
    vector = q[..., :3]
    scalar = q[..., 3:4]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, scalar)
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1e-12,
    )
    return vector * scale


def quaternion_xyzw_from_rotvec(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.divide(
        np.sin(half),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > 1e-12,
    )
    return np.concatenate([r * scale, np.cos(half)], axis=-1)


def quaternion_geodesic_degrees(q_true: np.ndarray, q_pred: np.ndarray) -> np.ndarray:
    """Geodesic SO(3) angle, invariant to quaternion sign."""
    a = np.asarray(q_true, dtype=np.float64)
    b = np.asarray(q_pred, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-15)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-15)
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = b[..., :3], b[..., 3:4]
    # conjugate(a) * b, both in xyzw ordering
    rv = aw * bv - bw * av - np.cross(av, bv)
    rw = aw * bw + np.sum(av * bv, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(np.linalg.norm(rv, axis=-1), np.abs(rw[..., 0]))
    return np.degrees(angle)


def inject_optional_author_dependencies() -> bool:
    """Provide only the two unused/elementary imports missing in a CPU env.

    The author model code imports PyTorch3D transforms and THOP at module load.
    This lightweight compatibility module implements the exact quaternion
    formulas needed by the released models; THOP profiling is never called.
    Returns True if the transform shim was installed.
    """
    shimmed = False
    if importlib.util.find_spec("pytorch3d") is None:
        import torch

        p3d = types.ModuleType("pytorch3d")
        transforms = types.ModuleType("pytorch3d.transforms")

        def quaternion_to_axis_angle(q_wxyz: Any) -> Any:
            vector = q_wxyz[..., 1:]
            scalar = q_wxyz[..., :1]
            norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
            angle = 2.0 * torch.atan2(norm, scalar)
            scale = angle / norm.clamp_min(1e-12)
            scale = torch.where(norm > 1e-12, scale, torch.full_like(scale, 2.0))
            return vector * scale

        def axis_angle_to_quaternion(r: Any) -> Any:
            angle = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
            half = 0.5 * angle
            scale = torch.sin(half) / angle.clamp_min(1e-12)
            scale = torch.where(angle > 1e-12, scale, torch.full_like(scale, 0.5))
            return torch.cat([torch.cos(half), r * scale], dim=-1)

        transforms.quaternion_to_axis_angle = quaternion_to_axis_angle
        transforms.axis_angle_to_quaternion = axis_angle_to_quaternion
        p3d.transforms = transforms
        sys.modules["pytorch3d"] = p3d
        sys.modules["pytorch3d.transforms"] = transforms
        shimmed = True
    if importlib.util.find_spec("thop") is None:
        thop = types.ModuleType("thop")
        thop.profile = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("THOP profiling is not used by this benchmark")
        )
        sys.modules["thop"] = thop
    return shimmed


@dataclass
class Record:
    name: str
    split: str
    trajectory: str
    run: int
    x: np.ndarray
    u: np.ndarray
    raw_sha256: str


def load_record(path: Path, entry: dict[str, Any]) -> Record:
    df = pd.read_csv(path)
    q = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float32)
    rotvec = rotvec_from_quaternion_xyzw(q)
    x = np.column_stack(
        [
            df[["x", "y", "z"]].to_numpy(dtype=np.float64),
            df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64),
            rotvec,
            df[["wx", "wy", "wz"]].to_numpy(dtype=np.float64),
        ]
    )
    u = df[["m1_rads", "m2_rads", "m3_rads", "m4_rads"]].to_numpy(
        dtype=np.float64
    )
    if len(df) != entry["rows"] or x.shape[1] != 12 or u.shape[1] != 4:
        raise ValueError(f"Unexpected row/state/input shape for {path}")
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError(f"Non-finite value in {path}")
    timestamp = df["t"].to_numpy(dtype=np.float64)
    dt = np.diff(timestamp)
    if not np.allclose(dt, DT, atol=2.1e-11, rtol=0.0):
        raise ValueError(f"Unexpected sample interval in {path}")
    return Record(
        name=path.name,
        split=entry["split"],
        trajectory=entry["trajectory"],
        run=int(path.stem.rsplit("run", 1)[1]),
        x=x,
        u=u,
        raw_sha256=sha256(path),
    )


def load_records(author_root: Path, data_root: Path, manifest_path: Path) -> list[Record]:
    manifest = json.loads(manifest_path.read_text())
    expected_commit = "2d921b57d166fe2debe08a5d39bd07297c5abc39"
    if manifest["source_commit"] != expected_commit:
        raise ValueError("Unexpected Nano-Drone source commit in manifest")
    actual_commit = subprocess.run(
        ["git", "-C", str(author_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise ValueError(f"Author repository commit mismatch: {actual_commit}")
    records = []
    for entry in manifest["files"]:
        pointer = author_root / entry["path"]
        pointer_text = pointer.read_text(errors="replace")
        if "version https://git-lfs.github.com/spec/v1" not in pointer_text:
            raise ValueError(f"Expected pinned LFS pointer at {pointer}")
        pointer_oid = next(
            (line.split(":", 1)[1] for line in pointer_text.splitlines() if line.startswith("oid sha256:")),
            None,
        )
        pointer_size = next(
            (int(line.split()[1]) for line in pointer_text.splitlines() if line.startswith("size ")),
            None,
        )
        path = data_root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing external CSV object: {path}")
        if pointer_oid != entry["lfs_sha256"] or pointer_size != entry["size_bytes"]:
            raise ValueError(f"Tracked pointer differs from manifest: {pointer}")
        if sha256(path) != pointer_oid or path.stat().st_size != pointer_size:
            raise ValueError(f"Raw CSV hash/size mismatch: {path}")
        records.append(load_record(path, entry))
    return records


def window_arrays(record: Record, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = record.x[starts]
    u_seq = np.stack([record.u[t : t + HORIZON] for t in starts])
    truth = np.stack([record.x[t + 1 : t + HORIZON + 1] for t in starts])
    return x0, u_seq, truth


def nonoverlap_starts(record: Record, first: int = 0) -> np.ndarray:
    # x[t] plus x[t+1:t+HORIZON+1] requires t+HORIZON < len(x).
    return np.arange(first, len(record.x) - HORIZON, OBSERVATION_STEP, dtype=int)


def make_polynomial_library(z: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    columns = [np.ones(len(z), dtype=np.float64)]
    columns.extend(z[:, i] for i in range(z.shape[1]))
    columns.extend(z[:, i] * z[:, j] for i, j in pairs)
    return np.column_stack(columns)


class ARXModel:
    def __init__(self, x_mean: np.ndarray, x_scale: np.ndarray, u_mean: np.ndarray, u_scale: np.ndarray, coef: np.ndarray, alpha: float):
        self.x_mean, self.x_scale = x_mean, x_scale
        self.u_mean, self.u_scale = u_mean, u_scale
        self.coef, self.alpha = coef, float(alpha)
        self.kind = "arx"

    def predict_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xn = (x - self.x_mean) / self.x_scale
        un = (u - self.u_mean) / self.u_scale
        reg = np.column_stack([np.ones(len(xn)), xn, un])
        yn = reg @ self.coef
        return yn * self.x_scale + self.x_mean


class SINDYcModel:
    def __init__(self, x_mean: np.ndarray, x_scale: np.ndarray, u_mean: np.ndarray, u_scale: np.ndarray, phi_mean: np.ndarray, phi_scale: np.ndarray, coef: np.ndarray, threshold: float, pairs: list[tuple[int, int]]):
        self.x_mean, self.x_scale = x_mean, x_scale
        self.u_mean, self.u_scale = u_mean, u_scale
        self.phi_mean, self.phi_scale = phi_mean, phi_scale
        self.coef, self.threshold, self.pairs = coef, float(threshold), pairs
        self.kind = "sindyc"

    def predict_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xn = (x - self.x_mean) / self.x_scale
        un = (u - self.u_mean) / self.u_scale
        raw = make_polynomial_library(np.column_stack([xn, un]), self.pairs)
        phi = (raw - self.phi_mean) / self.phi_scale
        derivative = phi @ self.coef
        next_norm = xn + DT * derivative
        return next_norm * self.x_scale + self.x_mean


def prepare_source_normalization(records: list[Record], scaler_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_scaler = joblib.load(scaler_dir / "x_scaler.pkl")
    u_scaler = joblib.load(scaler_dir / "u_scaler.pkl")
    x_scale = np.maximum(np.asarray(x_scaler.scale_, dtype=np.float64), 1e-12)
    u_scale = np.maximum(np.asarray(u_scaler.scale_, dtype=np.float64), 1e-12)
    return (
        np.asarray(x_scaler.mean_, dtype=np.float64),
        x_scale,
        np.asarray(u_scaler.mean_, dtype=np.float64),
        u_scale,
    )


def fit_arx(source: list[Record], x_mean: np.ndarray, x_scale: np.ndarray, u_mean: np.ndarray, u_scale: np.ndarray, alpha: float) -> ARXModel:
    xs, us, ys = [], [], []
    for record in source:
        xs.append((record.x[:-1] - x_mean) / x_scale)
        us.append((record.u[:-1] - u_mean) / u_scale)
        ys.append((record.x[1:] - x_mean) / x_scale)
    x = np.concatenate(xs)
    u = np.concatenate(us)
    y = np.concatenate(ys)
    design = np.column_stack([np.ones(len(x)), x, u])
    gram = design.T @ design
    rhs = design.T @ y
    penalty = np.eye(gram.shape[0]) * alpha
    penalty[0, 0] = 0.0
    coef = solve(gram + penalty, rhs, assume_a="pos", check_finite=False)
    return ARXModel(x_mean, x_scale, u_mean, u_scale, coef, alpha)


def fit_sindyc_candidates(source: list[Record], x_mean: np.ndarray, x_scale: np.ndarray, u_mean: np.ndarray, u_scale: np.ndarray) -> tuple[list[SINDYcModel], dict[str, Any]]:
    z_parts, y_parts = [], []
    for record in source:
        x = (record.x[:-1] - x_mean) / x_scale
        u = (record.u[:-1] - u_mean) / u_scale
        x_next = (record.x[1:] - x_mean) / x_scale
        z_parts.append(np.column_stack([x, u]))
        y_parts.append((x_next - x) / DT)
    z = np.concatenate(z_parts)
    target = np.concatenate(y_parts)
    pairs = [(i, j) for i in range(z.shape[1]) for j in range(i, z.shape[1])]
    raw = make_polynomial_library(z, pairs)
    phi_mean = raw.mean(axis=0)
    phi_scale = raw.std(axis=0)
    # Preserve the constant basis function as an intercept. Centering the
    # all-ones column would silently remove the SINDYc bias term.
    phi_mean[0] = 0.0
    phi_scale[0] = 1.0
    phi_scale[phi_scale < 1e-12] = 1.0
    phi = (raw - phi_mean) / phi_scale
    gram = phi.T @ phi
    rhs = phi.T @ target
    ridge = 1e-8
    base = solve(gram + ridge * np.eye(gram.shape[0]), rhs, assume_a="pos", check_finite=False)
    models = []
    term_counts = {}
    for threshold in SINDY_THRESHOLD_GRID:
        coef = base.copy()
        active = np.abs(coef) >= threshold
        active[0, :] = True
        for _ in range(10):
            next_coef = np.zeros_like(coef)
            next_active = active.copy()
            for output in range(target.shape[1]):
                ids = np.flatnonzero(active[:, output])
                if not len(ids):
                    ids = np.array([0])
                local_gram = gram[np.ix_(ids, ids)] + ridge * np.eye(len(ids))
                next_coef[ids, output] = solve(
                    local_gram, rhs[ids, output], assume_a="pos", check_finite=False
                )
                next_active[:, output] = np.abs(next_coef[:, output]) >= threshold
                next_active[0, output] = True
            if np.array_equal(active, next_active):
                coef = next_coef
                break
            active, coef = next_active, next_coef
        term_counts[str(threshold)] = [int(np.count_nonzero(coef[:, j])) for j in range(coef.shape[1])]
        models.append(SINDYcModel(x_mean, x_scale, u_mean, u_scale, phi_mean, phi_scale, coef, threshold, pairs))
    singular_values = np.linalg.svd(phi, compute_uv=False)
    tol = singular_values[0] * max(phi.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > tol))
    diagnostics = {
        "library_features": int(phi.shape[1]),
        "effective_rank": rank,
        "condition_number_nonzero_singular_values": float(singular_values[0] / singular_values[rank - 1]) if rank else None,
        "smallest_singular_values": singular_values[-8:].tolist(),
        "active_terms_per_state_by_threshold": term_counts,
    }
    return models, diagnostics


def load_author_models(author_root: Path) -> tuple[dict[str, Any], bool, dict[str, str]]:
    shimmed = inject_optional_author_dependencies()
    import torch

    expected_commit = "2d921b57d166fe2debe08a5d39bd07297c5abc39"
    model_file = verify_tracked_file_at_commit(author_root, "models/models.py", expected_commit)
    spec = importlib.util.spec_from_file_location("nanodrone_author_models", model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load author model module: {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    scaler_root = author_root / "scalers"
    bundles: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    configs = {
        "author_mlp": ("residual_random_square_chirp", "residual_random_square_chirp.pt"),
        "author_lstm": ("lstm_random_square_chirp", "lstm_random_square_chirp.pt"),
        "author_hybrid": ("phys+res_random_square_chirp", "phys+res_random_square_chirp.pt"),
    }
    for label, (scaler_name, checkpoint_name) in configs.items():
        scaler_dir = scaler_root / scaler_name
        x_scaler_path = verify_tracked_file_at_commit(
            author_root, f"scalers/{scaler_name}/x_scaler.pkl", expected_commit
        )
        u_scaler_path = verify_tracked_file_at_commit(
            author_root, f"scalers/{scaler_name}/u_scaler.pkl", expected_commit
        )
        x_scaler = joblib.load(x_scaler_path)
        u_scaler = joblib.load(u_scaler_path)
        checkpoint = verify_tracked_file_at_commit(
            author_root, f"out/models/{checkpoint_name}", expected_commit
        )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if label == "author_mlp":
            model = module.ResidualQuadModel(**ckpt["config"])
        elif label == "author_lstm":
            model = module.QuadLSTM(**ckpt["config"])
        else:
            params = {
                "g": 9.81,
                "m": 0.045,
                "J": torch.diag(torch.tensor([2.3951e-5, 2.3951e-5, 3.2347e-6])),
                "thrust_to_weight": 2.0,
                "max_torque": torch.tensor([1e-2, 1e-2, 3e-3]),
            }
            phys = module.PhysQuadModel(params, DT)
            residual = module.ResidualQuadModel(
                hidden_dim=ckpt["config"]["hidden_dim"],
                num_layers=ckpt["config"]["num_layers"],
                dt=DT,
            )
            model = module.PhysResQuadModel(phys, residual, x_scaler, u_scaler)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        bundles[label] = {
            "kind": label,
            "model": model,
            "x_mean": np.asarray(x_scaler.mean_, dtype=np.float64),
            "x_scale": np.asarray(x_scaler.scale_, dtype=np.float64),
            "u_mean": np.asarray(u_scaler.mean_, dtype=np.float64),
            "u_scale": np.asarray(u_scaler.scale_, dtype=np.float64),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "checkpoint_val_loss": float(ckpt.get("val_loss", math.nan)),
            "checkpoint_sha256": sha256(checkpoint),
            "scaler_x_sha256": sha256(x_scaler_path),
            "scaler_u_sha256": sha256(u_scaler_path),
        }
        hashes[checkpoint_name] = bundles[label]["checkpoint_sha256"]
    # The released physical model has no trained checkpoint; retain the
    # as-released implementation as a diagnostic baseline.
    params = {
        "g": 9.81,
        "m": 0.045,
        "J": torch.diag(torch.tensor([2.3951e-5, 2.3951e-5, 3.2347e-6])),
        "thrust_to_weight": 2.0,
        "max_torque": torch.tensor([1e-2, 1e-2, 3e-3]),
    }
    physical = module.PhysQuadModel(params, DT)
    physical.eval()
    bundles["author_physics"] = {"kind": "author_physics", "model": physical}
    return bundles, shimmed, hashes


def predict_windows(model: Any, x0: np.ndarray, u_seq: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Roll out a window, then apply one common path-displacement multiplier."""
    kind = model.get("kind", "") if isinstance(model, dict) else getattr(model, "kind", "")
    if kind.startswith("author"):
        import torch

        net = model["model"]
        with torch.no_grad():
            if kind == "author_physics":
                state = torch.as_tensor(x0, dtype=torch.float32)
                controls = torch.as_tensor(u_seq, dtype=torch.float32)
                predictions = []
                for step in range(HORIZON):
                    state = net.one_step(state, controls[:, step, :])
                    predictions.append(state.unsqueeze(1))
                path = torch.cat(predictions, dim=1).cpu().numpy().astype(np.float64)
            else:
                x_mean = torch.as_tensor(model["x_mean"], dtype=torch.float32)
                x_scale = torch.as_tensor(model["x_scale"], dtype=torch.float32)
                u_mean = torch.as_tensor(model["u_mean"], dtype=torch.float32)
                u_scale = torch.as_tensor(model["u_scale"], dtype=torch.float32)
                x0n = (torch.as_tensor(x0, dtype=torch.float32) - x_mean) / x_scale
                usn = (torch.as_tensor(u_seq, dtype=torch.float32) - u_mean) / u_scale
                if kind == "author_lstm":
                    predn = net(x0n, usn)
                else:
                    state = x0n
                    predictions = []
                    for step in range(HORIZON):
                        state = net.one_step(state, usn[:, step, :])
                        predictions.append(state.unsqueeze(1))
                    predn = torch.cat(predictions, dim=1)
                path = (predn * x_scale + x_mean).cpu().numpy().astype(np.float64)
    else:
        state = np.asarray(x0, dtype=np.float64).copy()
        predictions = []
        for step in range(HORIZON):
            state = model.predict_step(state, u_seq[:, step, :])
            predictions.append(state.copy())
        path = np.stack(predictions, axis=1)
    # The same scalar operation is applied to every model in the common
    # 12-D source-normalized state representation; it does not alter the
    # autoregressive path used to generate subsequent predictions.
    return x0[:, None, :] + float(gamma) * (path - x0[:, None, :])


def score_windows(pred: np.ndarray, truth: np.ndarray, x_scale: np.ndarray) -> dict[int, dict[str, float]]:
    output = {}
    for horizon in (1, 10, 50):
        err = pred[:, horizon - 1, :] - truth[:, horizon - 1, :]
        pred_state = pred[:, horizon - 1, :]
        true_state = truth[:, horizon - 1, :]
        finite_rows = np.isfinite(pred_state).all(axis=1)
        q_pred = quaternion_xyzw_from_rotvec(pred_state[:, 6:9])
        q_true = quaternion_xyzw_from_rotvec(true_state[:, 6:9])
        angle = quaternion_geodesic_degrees(q_true, q_pred)
        if finite_rows.all():
            state_nrmse = float(np.sqrt(np.mean((err / x_scale) ** 2)))
            position_rmse = float(np.sqrt(np.mean(np.sum(err[:, 0:3] ** 2, axis=1))))
            velocity_rmse = float(np.sqrt(np.mean(np.sum(err[:, 3:6] ** 2, axis=1))))
            orientation_mean = float(np.mean(angle))
            orientation_rmse = float(np.sqrt(np.mean(angle**2)))
            angular_velocity_rmse = float(np.sqrt(np.mean(np.sum(err[:, 9:12] ** 2, axis=1))))
        else:
            # Keep failures visible in the primary scores; finite-only
            # aggregation would silently reward unstable models.
            state_nrmse = math.inf
            position_rmse = math.inf
            velocity_rmse = math.inf
            orientation_mean = math.inf
            orientation_rmse = math.inf
            angular_velocity_rmse = math.inf
        output[horizon] = {
            "state_nrmse": state_nrmse,
            "position_rmse_m": position_rmse,
            "velocity_rmse_m_s": velocity_rmse,
            "orientation_geodesic_mean_deg": orientation_mean,
            "orientation_geodesic_rmse_deg": orientation_rmse,
            "angular_velocity_rmse_rad_s": angular_velocity_rmse,
            "nonfinite_fraction": float(1.0 - np.isfinite(pred[:, horizon - 1, :]).all(axis=1).mean()),
        }
    return output


def model_envelope_fraction(pred: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    finite = np.isfinite(pred)
    if not finite.any():
        return 1.0
    outside = (pred < low[None, None, :]) | (pred > high[None, None, :])
    return float(np.logical_and(outside, finite).sum() / finite.sum())


def validation_score(model: Any, records: list[Record], x_scale: np.ndarray) -> float:
    per_record = []
    for record in records:
        starts = nonoverlap_starts(record)
        x0, u_seq, truth = window_arrays(record, starts)
        pred = predict_windows(model, x0, u_seq, 1.0)
        if not np.isfinite(pred).all():
            return math.inf
        err = (pred[:, 49, :] - truth[:, 49, :]) / x_scale
        score = float(np.mean(err**2))
        if not np.isfinite(score):
            return math.inf
        per_record.append(score)
    return float(np.mean(per_record))


def fit_target_gamma(model: Any, record: Record, budget_seconds: int, x_scale: np.ndarray) -> tuple[float, int, float]:
    transition_count = budget_seconds * 100 // OBSERVATION_STEP
    if transition_count == 0:
        return 1.0, 0, math.nan
    starts = np.arange(0, transition_count * OBSERVATION_STEP, OBSERVATION_STEP, dtype=int)
    x0, u_seq, truth = window_arrays(record, starts)
    best_gamma, best_loss = 1.0, math.inf
    for gamma in GAMMA_GRID:
        pred = predict_windows(model, x0, u_seq, float(gamma))
        if not np.isfinite(pred).all():
            continue
        err = (pred[:, 49, :] - truth[:, 49, :]) / x_scale
        loss = float(np.mean(err**2))
        if np.isfinite(loss) and loss < best_loss:
            best_gamma, best_loss = float(gamma), loss
    return best_gamma, transition_count, best_loss


def evaluate_record(model_name: str, model: Any, record: Record, budget_seconds: int, gamma: float, x_scale: np.ndarray, source_low: np.ndarray, source_high: np.ndarray, adaptation: str) -> tuple[list[dict[str, Any]], float]:
    first = budget_seconds * 100
    starts = nonoverlap_starts(record, first=first)
    x0, u_seq, truth = window_arrays(record, starts)
    pred = predict_windows(model, x0, u_seq, gamma)
    metrics = score_windows(pred, truth, x_scale)
    rows = []
    for horizon, values in metrics.items():
        rows.append({
            "model": model_name,
            "adaptation": adaptation,
            "budget_seconds": budget_seconds,
            "gamma": gamma,
            "run": record.run,
            "trajectory": record.trajectory,
            "prediction_horizon_steps": horizon,
            "prediction_horizon_seconds": horizon * DT,
            "n_windows": len(starts),
            **values,
            "outside_source_envelope_fraction": model_envelope_fraction(pred[:, :horizon, :], source_low, source_high),
        })
    return rows, float(len(starts))


def endpoint_interpolation(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Interpolate vector states linearly and attitude along the SO(3) geodesic."""
    fractions = np.arange(1, HORIZON + 1, dtype=np.float64)[None, :, None] / HORIZON
    result = x0[:, None, :] + fractions * (x1[:, None, :] - x0[:, None, :])

    q0 = quaternion_xyzw_from_rotvec(x0[:, 6:9])[:, None, :]
    q1 = quaternion_xyzw_from_rotvec(x1[:, 6:9])[:, None, :]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.divide(
        np.sin((1.0 - fractions) * theta),
        sin_theta,
        out=1.0 - fractions + np.zeros_like(theta),
        where=np.abs(sin_theta) > 1e-8,
    )
    w1 = np.divide(
        np.sin(fractions * theta),
        sin_theta,
        out=fractions + np.zeros_like(theta),
        where=np.abs(sin_theta) > 1e-8,
    )
    q_interp = w0 * q0 + w1 * q1
    q_interp /= np.maximum(np.linalg.norm(q_interp, axis=-1, keepdims=True), 1e-15)
    result[:, :, 6:9] = rotvec_from_quaternion_xyzw(q_interp)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, default=ROOT / "data/research_control_v1/external/nanodrone")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/research_control_v1/external/nanodrone_lfs")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/algorithmic-control-v1/nanodrone_data_manifest_20261103.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs/algorithmic-control-v1/results/nanodrone_external_v1")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(args.threads)

    records = load_records(args.author_root, args.data_root, args.manifest)
    train = [r for r in records if r.split == "train" and r.run in (1, 2, 3)]
    validation = [r for r in records if r.split == "train" and r.run == 4]
    test = [r for r in records if r.split == "test"]
    if len(train) != 9 or len(validation) != 3 or len(test) != 3:
        raise ValueError(f"Unexpected split sizes: train={len(train)}, val={len(validation)}, test={len(test)}")

    scaler_dir = args.author_root / "scalers" / "residual_random_square_chirp"
    x_mean, x_scale, u_mean, u_scale = prepare_source_normalization(train, scaler_dir)
    source_x = np.concatenate([r.x for r in train], axis=0)
    source_low, source_high = source_x.min(axis=0), source_x.max(axis=0)

    # Select low-order model hyperparameters only on the source run-4 split.
    arx_models = [fit_arx(train, x_mean, x_scale, u_mean, u_scale, alpha) for alpha in ARX_ALPHA_GRID]
    arx_scores = [validation_score(model, validation, x_scale) for model in arx_models]
    selected_arx = arx_models[best_finite_index(arx_scores, "ARX")]
    sindy_models, sindy_diag = fit_sindyc_candidates(train, x_mean, x_scale, u_mean, u_scale)
    sindy_scores = [validation_score(model, validation, x_scale) for model in sindy_models]
    selected_sindy = sindy_models[best_finite_index(sindy_scores, "SINDYc")]

    author_models, p3d_shimmed, checkpoint_hashes = load_author_models(args.author_root)
    models: dict[str, Any] = {
        "arx": selected_arx,
        "sindyc_degree2": selected_sindy,
        **author_models,
    }
    # Report the frozen source-run-4 ranking across every candidate. The
    # Melon target labels are not used for this source-only model choice.
    model_validation_scores = {
        name: validation_score(model, validation, x_scale)
        for name, model in models.items()
    }
    finite_models = [name for name, score in model_validation_scores.items() if np.isfinite(score)]
    if not finite_models:
        raise RuntimeError("All candidate models failed on source validation")
    source_selected_model = min(finite_models, key=model_validation_scores.__getitem__)

    # Source-only model diagnostics are held fixed and do not use Melon labels.
    source_records_for_design = train
    inputs = np.concatenate([r.u for r in source_records_for_design], axis=0)
    input_cov = np.cov(inputs, rowvar=False)
    input_eig = np.linalg.eigvalsh(input_cov)
    arx_design = np.column_stack([
        np.ones(sum(len(r.x) - 1 for r in train)),
        np.concatenate([(r.x[:-1] - x_mean) / x_scale for r in train]),
        np.concatenate([(r.u[:-1] - u_mean) / u_scale for r in train]),
    ])
    arx_singular = np.linalg.svd(arx_design, compute_uv=False)
    arx_tol = arx_singular[0] * max(arx_design.shape) * np.finfo(float).eps
    arx_rank = int(np.count_nonzero(arx_singular > arx_tol))
    excitation = {
        "training_run_count": len(train),
        "motor_speed_covariance_eigenvalues": input_eig.tolist(),
        "motor_speed_covariance_rank": int(np.linalg.matrix_rank(input_cov)),
        "motor_speed_covariance_condition_number": float(input_eig[-1] / input_eig[0]) if input_eig[0] > 0 else None,
        "arx_design_feature_count": int(arx_design.shape[1]),
        "arx_design_effective_rank": arx_rank,
        "arx_design_condition_number": float(arx_singular[0] / arx_singular[arx_rank - 1]) if arx_rank else None,
        "sindyc_degree2_library": sindy_diag,
    }

    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for record in test:
        for budget in TARGET_BUDGETS_SECONDS:
            for model_name, model in models.items():
                gamma, n_transitions, prefix_loss = fit_target_gamma(model, record, budget, x_scale)
                if budget == 0:
                    rows, _ = evaluate_record(
                        model_name, model, record, budget, 1.0, x_scale,
                        source_low, source_high, "source_frozen"
                    )
                    metric_rows.extend(rows)
                else:
                    frozen_rows, _ = evaluate_record(
                        model_name, model, record, budget, 1.0, x_scale,
                        source_low, source_high, "source_frozen"
                    )
                    adapted_rows, _ = evaluate_record(
                        model_name, model, record, budget, gamma, x_scale,
                        source_low, source_high, "target_scalar_gamma"
                    )
                    metric_rows.extend(frozen_rows)
                    metric_rows.extend(adapted_rows)
                calibration_rows.append({
                    "model": model_name,
                    "run": record.run,
                    "budget_seconds": budget,
                    "adaptation": "none" if budget == 0 else "target_scalar_gamma",
                    "target_prefix_transitions": n_transitions,
                    "gamma": 1.0 if budget == 0 else gamma,
                    "prefix_endpoint_nrmse": math.sqrt(prefix_loss) if np.isfinite(prefix_loss) else math.nan,
                })
            # Baselines are always evaluated at gamma=0 and gamma=1 from the
            # same sparse observation starts; the offline endpoint bridge is
            # scored separately and uses the true right endpoint.
            for base_name, gamma in (("zero_order_hold", 0.0),):
                rows, _ = evaluate_record(
                    base_name, selected_arx, record, budget, gamma, x_scale,
                    source_low, source_high, "zero_order_hold"
                )
                metric_rows.extend(rows)
            starts = nonoverlap_starts(record, first=budget * 100)
            x0, _, truth = window_arrays(record, starts)
            endpoint = endpoint_interpolation(x0, truth[:, 49, :])
            endpoint_metrics = score_windows(endpoint, truth, x_scale)
            for horizon, values in endpoint_metrics.items():
                metric_rows.append({
                    "model": "endpoint_linear_interpolation_offline",
                    "adaptation": "offline_endpoint",
                    "budget_seconds": budget,
                    "gamma": math.nan,
                    "run": record.run,
                    "trajectory": record.trajectory,
                    "prediction_horizon_steps": horizon,
                    "prediction_horizon_seconds": horizon * DT,
                    "n_windows": len(starts),
                    **values,
                    "outside_source_envelope_fraction": model_envelope_fraction(endpoint[:, :horizon, :], source_low, source_high),
                })

    metrics_path = args.out_dir / "nanodrone_metrics_by_run.csv"
    calibration_path = args.out_dir / "nanodrone_target_calibration.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(calibration_path, calibration_rows)
    metrics_df = pd.DataFrame(metric_rows)
    summary = (
        metrics_df.groupby(["model", "adaptation", "budget_seconds", "prediction_horizon_steps"], dropna=False)
        .agg(
            runs=("run", "nunique"),
            state_nrmse_mean=("state_nrmse", "mean"),
            state_nrmse_sd_between_runs=("state_nrmse", "std"),
            position_rmse_m_mean=("position_rmse_m", "mean"),
            velocity_rmse_m_s_mean=("velocity_rmse_m_s", "mean"),
            orientation_geodesic_mean_deg=("orientation_geodesic_mean_deg", "mean"),
            angular_velocity_rmse_rad_s_mean=("angular_velocity_rmse_rad_s", "mean"),
            nonfinite_fraction_mean=("nonfinite_fraction", "mean"),
            outside_source_envelope_fraction_mean=("outside_source_envelope_fraction", "mean"),
        )
        .reset_index()
    )
    summary_path = args.out_dir / "nanodrone_summary.csv"
    summary.to_csv(summary_path, index=False)
    excitation_path = args.out_dir / "nanodrone_excitation_diagnostics.json"
    excitation_path.write_text(
        json.dumps(clean_json(excitation), indent=2, allow_nan=False) + "\n"
    )
    validation_report = {
        "arx_alpha_grid": list(ARX_ALPHA_GRID),
        "arx_validation_normalized_mse_h50": arx_scores,
        "selected_arx_alpha": selected_arx.alpha,
        "sindyc_threshold_grid": list(SINDY_THRESHOLD_GRID),
        "sindyc_validation_normalized_mse_h50": sindy_scores,
        "selected_sindyc_threshold": selected_sindy.threshold,
        "candidate_model_validation_normalized_mse_h50": model_validation_scores,
        "source_validation_selected_model": source_selected_model,
        "source_model_selection_metric": "equal_mean_per_run_normalized_mse_at_h50",
    }
    source_validation_path = args.out_dir / "nanodrone_source_validation.json"
    source_validation_path.write_text(
        json.dumps(clean_json(validation_report), indent=2, allow_nan=False) + "\n"
    )
    protocol_path = ROOT / "docs/algorithmic-control-v1/NANODRONE_EXTERNAL_PROTOCOL_V1_20261103.md"
    manifest_out = {
        "protocol": "NANODRONE_EXTERNAL_PROTOCOL_V1_20261103.md",
        "protocol_sha256": sha256(protocol_path),
        "benchmark_script_sha256": sha256(Path(__file__)),
        "source_repository_commit": "2d921b57d166fe2debe08a5d39bd07297c5abc39",
        "data_manifest_sha256": sha256(args.manifest),
        "author_models": {
            name: {
                "checkpoint_sha256": model.get("checkpoint_sha256"),
                "scaler_x_sha256": model.get("scaler_x_sha256"),
                "scaler_u_sha256": model.get("scaler_u_sha256"),
                "checkpoint_epoch": model.get("checkpoint_epoch"),
                "checkpoint_val_loss": model.get("checkpoint_val_loss"),
            }
            for name, model in author_models.items()
        },
        "author_checkpoint_hashes": checkpoint_hashes,
        "author_physics_baseline_limitation": "As-released PhysQuad sets omega_dot to zero; it is retained only as an implementation diagnostic.",
        "pytorch3d_compatibility_shim_used": p3d_shimmed,
        "model_metrics_sha256": sha256(metrics_path),
        "calibration_metrics_sha256": sha256(calibration_path),
        "summary_sha256": sha256(summary_path),
        "excitation_diagnostics_sha256": sha256(excitation_path),
        "source_validation_sha256": sha256(source_validation_path),
        "split_counts": {"train_source_runs": len(train), "source_validation_runs": len(validation), "target_melon_runs": len(test)},
        "source_validation_selected_model": source_selected_model,
        "target_budgets_seconds": list(TARGET_BUDGETS_SECONDS),
        "target_observation_step_seconds": OBSERVATION_STEP * DT,
        "scoring_horizons_steps": [1, 10, 50],
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "torch": torch.__version__,
        },
        "scaler_pickle_source_version": "1.7.2",
        "scaler_pickle_runtime_version_matches": sklearn.__version__ == "1.7.2",
        "scaler_pickle_version_warning": None if sklearn.__version__ == "1.7.2" else "Author StandardScaler pickles were saved with scikit-learn 1.7.2; this run uses the runtime version recorded above. Only fitted mean_ and scale_ arrays are read.",
    }
    (args.out_dir / "nanodrone_run_manifest.json").write_text(
        json.dumps(clean_json(manifest_out), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": "NANODRONE_EXTERNAL_BENCHMARK_COMPLETE",
        "output_dir": str(args.out_dir),
        "rows_scored": len(metric_rows),
        "selected_arx_alpha": selected_arx.alpha,
        "selected_sindyc_threshold": selected_sindy.threshold,
        "pytorch3d_compatibility_shim_used": p3d_shimmed,
    }, indent=2))


if __name__ == "__main__":
    main()
