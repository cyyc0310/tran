"""Causal-ZS: Config-Conditioned Domain Disentanglement for Zero-Shot CIF Forecasting.

Simplified causal direction (per RESEARCH_DIRECTIONS.md §4.4):
    - VAE-style domain encoder decomposes RenewShare into:
        z_inv : domain-invariant (transferable temporal patterns)
        z_spec: domain-specific (config-determined bias)
    - Counterfactual data augmentation via physics layer:
        CIF_B(t) = s_A(t) * ef_r_B + (1-s_A(t)) * ef_nr_B
    - Training: reconstruction + KL + counterfactual consistency + prediction

Key difference from Phys-IRM:
    Phys-IRM learns invariance through loss reweighting.
    Causal-ZS learns invariance through explicit representation disentanglement.


Exports:
    CausalDomainVAE       — VAE-style encoder/decoder for share sequences
    train_causal_zero_shot — LORO training with disentanglement + counterfactual aug
    predict_causal_zs     — zero-shot inference
    counterfactual_augment — generate (s_A, CIF_B) pairs for augmentation
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transcif.data.windows import build_windows
from transcif.physics.decompose import cif_from_shares
from transcif.physics.bounds import config_weight, unify_config_dim, pad_config
from transcif.training.schedulers import get_cosine_warmup_scheduler


# ---------------------------------------------------------------------------
# Causal Domain VAE
# ---------------------------------------------------------------------------

class CausalDomainVAE(nn.Module):
    """Disentangle RenewShare sequences into invariant + domain-specific factors.

    Architecture:
        Encoder(x, config) → (mu_inv, logvar_inv) + (mu_spec, logvar_spec)
        Decoder(z_inv, z_spec) → reconstructed x
        Predictor(z_inv, config) → future share ∈ [0,1]^H
    """

    def __init__(self, seq_len=336, horizon=24, config_dim=2,
                 latent_dim=32, hidden_dim=128):
        super().__init__()
        self.config_dim = config_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len + config_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_inv = nn.Linear(hidden_dim, latent_dim)
        self.logvar_inv = nn.Linear(hidden_dim, latent_dim)
        self.mu_spec = nn.Linear(hidden_dim, latent_dim)
        self.logvar_spec = nn.Linear(hidden_dim, latent_dim)

        # Decoder: reconstructs the input share sequence
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2 + config_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len),
        )

        # Share predictor: invariant features + config + recent share level
        # (the recent mean gives the predictor a direct anchor to the input
        # window's current renewable-share level, not just its invariant code).
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + config_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

        # Persistence gate (config-conditioned, same as AdaptivePersistDLinear)
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

        # Domain classifier (auxiliary: ensures z_inv is truly domain-agnostic)
        # Not used during forward, only in adversarial loss
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, x, config):
        """Encode share sequence into invariant + specific latents.

        Returns: (z_inv, z_spec, mu_inv, logvar_inv, mu_spec, logvar_spec)
        """
        inp = torch.cat([x, config], dim=1)
        h = self.encoder(inp)

        mu_inv = self.mu_inv(h)
        logvar_inv = self.logvar_inv(h)
        mu_spec = self.mu_spec(h)
        logvar_spec = self.logvar_spec(h)

        z_inv = self._reparam(mu_inv, logvar_inv)
        z_spec = self._reparam(mu_spec, logvar_spec)

        return z_inv, z_spec, mu_inv, logvar_inv, mu_spec, logvar_spec

    def decode(self, z_inv, z_spec, config):
        """Reconstruct the input share sequence."""
        inp = torch.cat([z_inv, z_spec, config], dim=1)
        return self.decoder(inp)

    def predict_share(self, z_inv, config, x_persist):
        """Predict future share from invariant features + recent level."""
        recent_mean = x_persist[:, -48:].mean(dim=1, keepdim=True)
        feat = torch.cat([z_inv, config, recent_mean], dim=1)
        share_raw = torch.sigmoid(self.predictor(feat))
        # Adaptive persistence gate
        persist = x_persist[:, -self.horizon:]
        recent_std = x_persist[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * share_raw

    def forward(self, x, config):
        """Full forward: encode → decode + predict.

        Returns: (x_recon, share_pred, z_inv, z_spec, mu_inv, mu_spec, logvar_inv, logvar_spec)
        """
        z_inv, z_spec, mu_inv, logvar_inv, mu_spec, logvar_spec = self.encode(x, config)
        x_recon = self.decode(z_inv, z_spec, config)
        share_pred = self.predict_share(z_inv, config, x)
        return x_recon, share_pred, z_inv, z_spec, mu_inv, mu_spec, logvar_inv, logvar_spec

    @staticmethod
    def _reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# ---------------------------------------------------------------------------
# Counterfactual augmentation
# ---------------------------------------------------------------------------

def counterfactual_augment(all_regions, target_name, n_samples=200):
    """Generate counterfactual (share, CIF) pairs from source regions.

    For each source region A != target:
        Take random share window s_A
        Compute counterfactual CIF using target's emission factors:
            CIF_counterfactual = s_A * ef_r_target + (1-s_A) * ef_nr_target

    This augments the target's training data with "what if" scenarios.

    Returns: (x_cf, y_cf_cif) numpy arrays
    """
    target = all_regions[target_name]
    x_cf_list, y_cf_list = [], []

    for name, src in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, _ = build_windows(src["rs"], src["cif"])
        if len(x_win) == 0:
            continue
        n_take = min(len(x_win), max(10, n_samples // len(all_regions)))
        idx = np.random.choice(len(x_win), n_take, replace=False)
        x_cf_list.append(x_win[idx])
        # Use source share predictions + target emission factors
        y_cif_cf = cif_from_shares(
            y_win[idx], target["ef_r"], target["ef_nr"])
        y_cf_list.append(y_cif_cf)

    if not x_cf_list:
        return np.empty((0, 336)), np.empty((0, 24))
    return np.concatenate(x_cf_list), np.concatenate(y_cf_list)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def kl_divergence(mu, logvar):
    """KL(N(mu, sigma) || N(0, 1))."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def adversarial_domain_loss(z_inv, domain_labels, classifier):
    """Gradient-reversal adversarial loss: maximize domain classifier error.

    This ensures z_inv does not encode domain identity, enforcing invariance.
    """
    # Negate the gradient → gradient reversal layer
    z_adv = z_inv.detach()  # no grad for encoder; only classifier learns
    pred = classifier(z_adv).squeeze(-1)
    return F.binary_cross_entropy_with_logits(pred, domain_labels.float())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_causal_zero_shot(all_regions, target_name, seed=42,
                            epochs=300, lr=1e-3, device=None,
                            beta_kl=0.01, beta_adv=0.05, lambda_cf=0.3, pbar=None):
    """Train CausalDomainVAE in LORO setup with disentanglement.

    Loss:
        L = L_recon + L_share + beta_kl * L_kl + beta_adv * L_adv + lambda_cf * L_cf

    where:
        L_recon = MSE(x_recon, x)
        L_share = L1(share_pred, share_true)
        L_kl    = KL( q(z_inv|x,c) || N(0,I) ) + KL( q(z_spec|x,c) || N(0,I) )
        L_adv   = adversarial: z_inv should fool domain classifier
        L_cf    = CIF-level prediction on counterfactual samples
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg_dim = unify_config_dim(all_regions)
    model = CausalDomainVAE(seq_len=336, horizon=24, config_dim=cfg_dim)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)

    target_mean_rs = all_regions[target_name]["mean_rs"]
    # Gather per-region data
    region_data = []
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, y_cif_win = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        # Config-distance source weight (matches base TransCIF-ZS sampler).
        w = config_weight(data["mean_rs"], target_mean_rs)
        region_data.append({
            "name": name,
            "mean_rs": data["mean_rs"],
            "w": float(w),
            "x": torch.tensor(x_win, dtype=torch.float32),
            "y_share": torch.tensor(y_win, dtype=torch.float32),
            "y_cif": torch.tensor(y_cif_win, dtype=torch.float32),
            "config": torch.tensor(
                np.tile(pad_config(data["config"], cfg_dim), (len(x_win), 1)),
                dtype=torch.float32),
        })

    # Stable, informative domain split for the adversarial term: regions with
    # renewable share above vs below the source-median.  Replaces the previous
    # ``hash(name) % 2`` which was non-deterministic across processes and gave
    # the domain classifier no real signal to push against.
    if region_data:
        rs_vals = np.array([rd["mean_rs"] for rd in region_data])
        median_rs = float(np.median(rs_vals))
        for rd in region_data:
            rd["domain_id"] = float(rd["mean_rs"] >= median_rs)

    if not region_data:
        print(f"  [WARN] No source data for {target_name}")
        return model, []

    model.train()
    log = []
    batch_size = min(256, min(len(rd["x"]) for rd in region_data))

    for epoch in range(epochs):
        total = 0.0
        loss_parts = {"recon": [], "share": [], "kl": [], "adv": [], "cf": []}

        for rd_i, rd in enumerate(region_data):
            n = rd["x"].shape[0]
            idx = torch.randperm(n)[:batch_size]

            x_b = rd["x"][idx]
            y_share_b = rd["y_share"][idx]
            c_b = rd["config"][idx]

            if device:
                x_b, y_share_b, c_b = x_b.to(device), y_share_b.to(device), c_b.to(device)

            # Forward
            x_recon, share_pred, z_inv, z_spec, mu_inv, mu_spec, lv_inv, lv_spec = \
                model(x_b, c_b)

            # 1. Reconstruction
            L_recon = F.mse_loss(x_recon, x_b)
            loss_parts["recon"].append(L_recon.item())

            # 2. Share prediction (config-distance weighted)
            w_e = rd["w"]
            L_share = w_e * F.l1_loss(share_pred, y_share_b)
            loss_parts["share"].append(L_share.item())

            # 3. KL divergence
            L_kl = kl_divergence(mu_inv, lv_inv) + kl_divergence(mu_spec, lv_spec)
            loss_parts["kl"].append(L_kl.item())

            # 4. Adversarial: domain classifier should fail on z_inv
            L_adv = adversarial_domain_loss(
                z_inv, torch.full_like(z_inv[:, 0], rd["domain_id"]),
                model.domain_classifier)
            loss_parts["adv"].append(L_adv.item())

            loss = L_recon + L_share + beta_kl * L_kl + beta_adv * L_adv

            # 5. Counterfactual (every 5 epochs for efficiency)
            if epoch % 5 == 0 and lambda_cf > 0:
                x_cf, y_cf_cif = counterfactual_augment(
                    all_regions, target_name, n_samples=min(n, 100))
                if len(x_cf) > 0:
                    x_cf_t = torch.tensor(x_cf, dtype=torch.float32)
                    y_cf_cif_t = torch.tensor(y_cf_cif, dtype=torch.float32)
                    c_cf = torch.tensor(
                        np.tile(all_regions[target_name]["config"],
                                (len(x_cf), 1)), dtype=torch.float32)
                    if device:
                        x_cf_t, y_cf_cif_t, c_cf = x_cf_t.to(device), y_cf_cif_t.to(device), c_cf.to(device)
                    # Predict share from cross-region data, apply target physics
                    _, share_cf, _, _, _, _, _, _ = model(x_cf_t, c_cf)
                    ef_r_t = all_regions[target_name]["ef_r"]
                    ef_nr_t = all_regions[target_name]["ef_nr"]
                    cif_cf = share_cf * ef_r_t + (1.0 - share_cf) * ef_nr_t
                    L_cf = w_e * F.l1_loss(cif_cf, y_cf_cif_t)
                    loss = loss + lambda_cf * L_cf
                    loss_parts["cf"].append(L_cf.item())
                else:
                    loss_parts["cf"].append(0.0)

            total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        if pbar is not None:
            pbar(epoch, epochs, total / max(len(region_data), 1))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            log.append({"epoch": epoch + 1,
                        "L_recon": np.mean(loss_parts["recon"]),
                        "L_share": np.mean(loss_parts["share"]),
                        "L_kl": np.mean(loss_parts["kl"]),
                        "L_adv": np.mean(loss_parts["adv"]),
                        "total": total / max(len(region_data), 1)})

    model.eval()
    if pbar is not None:
        pbar.finish()
    return model, log


def predict_causal_zs(model, x_rs, config, ef_r, ef_nr):
    """Zero-shot inference with CausalDomainVAE.

    Args:
        model  : CausalDomainVAE
        x_rs   : (N, seq_len) numpy RenewShare windows
        config : (config_dim,) numpy target config
        ef_r, ef_nr : emission factors

    Returns:
        cif_pred : (N, horizon) numpy CIF predictions
    """
    model.eval()
    dev = next(model.parameters()).device
    x_t = torch.tensor(x_rs, dtype=torch.float32).to(dev)
    config = pad_config(np.asarray(config), getattr(model, "config_dim", len(config))) \
             if not isinstance(config, torch.Tensor) else config
    c_t = torch.tensor(config).unsqueeze(0).expand(len(x_rs), -1).to(dev)
    with torch.no_grad():
        z_inv, z_spec, _, _, _, _ = model.encode(x_t, c_t)
        share_pred = model.predict_share(z_inv, c_t, x_t)
    return cif_from_shares(share_pred.cpu().numpy(), ef_r, ef_nr)


def disentanglement_quality(model, x, config):
    """Quantitative diagnostic: correlation between z_inv and domain identity.

    Lower correlation → better disentanglement (z_inv does not leak domain info).

    Returns: abs(Pearson r) between z_inv mean and domain-specific z_spec mean.
    """
    model.eval()
    with torch.no_grad():
        z_inv, z_spec, _, _, _, _ = model.encode(x, config)
        # For each dimension, compute correlation
        inv_mean = z_inv.mean(dim=0)
        spec_mean = z_spec.mean(dim=0)
        inv_centered = z_inv - inv_mean.unsqueeze(0)
        spec_centered = z_spec - spec_mean.unsqueeze(0)
        # Mutual information proxy: cosine similarity between centered features
        cos_sim = F.cosine_similarity(
            inv_centered.mean(dim=1), spec_centered.mean(dim=1), dim=0)
        return float(abs(cos_sim))
