"""FuelDecompNet: physics-structured, config-conditioned zero-shot CIF model.

Architecture (TransCIF-FD, Phase FD-1):

    * per-fuel share heads with fuel-group inductive biases —
        solar   : astronomy envelope (clear-sky, exact for any date/coordinate)
                  × weather modulation × trailing/config level
        wind    : IEC power-curve transform of forecast wind speed, similarly
                  normalised against a trailing/config reference level
        baseload (nuclear/hydro/biomass/imports/other): slowly varying level +
                  a small calendar-driven correction
        thermal (coal/gas/petroleum): residual share 1 - Σ(non-dispatchable),
                  split by config-conditioned logits — dispatchable generation
                  fills the gap left by weather-driven units
    * exact physics layer CIF = Σ_f s_f · ef_f with a learned, bounded,
      config-conditioned EF correction (absorbs source-data methodology bias,
      e.g. the UK API's internal accounting)
    * aggregate renewable-share fallback head (DLinear + exog) that serves
      fuel-telemetry-free regions (AU) exactly like the legacy physics layer
      and doubles as an auxiliary consistency signal / ZS+ share_fn hook

Every conditioning input (fuel-share config, weather climatology, calendar,
astronomy) is deployment-public — the interface a telemetry-free target
region (a Chinese province with monthly fuel-mix statistics only) can supply.
"""

import torch
import torch.nn as nn

from transcif.data.fuel import (
    CANONICAL_FUELS, FUEL_INDEX, THERMAL_FUELS, BASELOAD_FUELS,
)

_N_FUEL = len(CANONICAL_FUELS)
_IDX_SOLAR = FUEL_INDEX["solar"]
_IDX_WIND = FUEL_INDEX["wind"]
_THERMAL_IDX = [FUEL_INDEX[f] for f in THERMAL_FUELS]
_BASELOAD_IDX = [FUEL_INDEX[f] for f in BASELOAD_FUELS]

# FD config layout (see data.fuel.FD_CONFIG_FIELDS)
_CFG_MEAN_RS = 0
_CFG_EF_NR = 1
_CFG_FUEL0 = 2                 # 10 fuel shares at [2:12]
_CFG_ANN_WINDCF = 12
_CFG_ANN_CSI = 13
_CFG_HAS_FUEL = 14


def _dyn_head(name, x, plain, gen):
    """Run a dynamic head: hypernet-generated per-sample weights when a
    generation dict is supplied, else the plain shared head."""
    if gen is not None:
        from transcif.models.hypernet import apply_generated_head  # noqa: PLC0415
        return apply_generated_head(x, name, gen)
    return plain(x)


class FuelDecompNet(nn.Module):
    """Physics-structured fuel-decomposed CIF forecaster.

    forward() returns ``(cif, shares)`` — CIF (B, H) in gCO2/kWh via the
    inline differentiable physics layer, and per-fuel shares (B, H, F).
    """

    def __init__(self, seq_len=336, horizon=24, n_weather=5, n_exog=10,
                 config_dim=16, hidden=32, use_hypernet=False):
        super().__init__()
        self.horizon = horizon
        self.n_weather = n_weather
        self.n_exog = n_exog
        self.use_hypernet = use_hypernet
        F = _N_FUEL

        # --- config encoder (small; capacity was shown harmful in Stage B/D)
        self.cfg_mlp = nn.Sequential(
            nn.Linear(config_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        # --- solar head: level × astronomy envelope × weather modulation
        self.solar_mod = nn.Linear(n_weather, 1)     # (B,H,5) -> (B,H,1) scalar per hour
        # --- wind head
        self.wind_mod = nn.Linear(n_weather, 1)
        # --- baseload heads: level (history or config) + masked exog-driven
        #     correction.  The config support mask keeps zero-share fuels at
        #     zero — a region with no imports must not hallucinate imports.
        #     (per-hour pointwise linear: (B,H,E) -> (B,H,5))
        self.base_delta = nn.Linear(n_exog, len(_BASELOAD_IDX))
        self.base_prior = nn.Linear(config_dim, len(_BASELOAD_IDX))
        # --- thermal split logits: log-config anchor + learned corrections.
        #     At zero-init the split equals the config thermal distribution.
        self.therm_cfg = nn.Linear(config_dim, len(_THERMAL_IDX))
        self.therm_dyn = nn.Linear(n_exog, len(_THERMAL_IDX))
        # --- learned bounded EF correction (multiplicative, per fuel);
        #     consumes the encoded config (hidden-dim), not the raw vector
        self.ef_corr = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, F),
        )
        # --- aggregate renewable-share head (AU path + auxiliary signal):
        #     DLinear trend/seasonal on rs history + config bias + exog term,
        #     mirroring the flagship AdaptivePersistDLinear pattern.
        self.rs_trend = nn.Linear(seq_len, horizon)
        self.rs_seasonal = nn.Linear(seq_len, horizon)
        self.rs_cfg_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        self.rs_exog = nn.Linear(n_exog, 1)
        # --- persistence gate on the aggregate head (history mode)
        self.rs_gate = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))
        # --- optional config hypernet (FD-2): config generates the weights
        #     of every per-hour dynamic head, zero-initialised so a fresh
        #     model still starts at the FD-1 physics prior.
        if use_hypernet:
            from transcif.models.hypernet import ConfigHyperNet  # noqa: PLC0415
            self.hypernet = ConfigHyperNet(config_dim=config_dim)

        self._init_small()

    def _init_small(self):
        """Zero-init the dynamic corrections so training starts at the
        physics/level prior instead of a random perturbation of it."""
        for m in (self.solar_mod, self.wind_mod, self.therm_dyn,
                  self.rs_exog):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.base_delta.weight)
        nn.init.zeros_(self.base_delta.bias)
        nn.init.zeros_(self.base_prior.weight)
        nn.init.zeros_(self.base_prior.bias)
        nn.init.zeros_(self.therm_cfg.weight)
        nn.init.zeros_(self.therm_cfg.bias)
        for m in self.ef_corr:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x_rs, x_fuel, x_weather, fut_weather, fut_exog,
                config, ef_vec, hist_mask=None):
        """Predict CIF and per-fuel shares for one batch of windows.

        Args:
            x_rs        (B, L)   renewable-share history
            x_fuel      (B, L, F) per-fuel share history (zeros when absent)
            x_weather   (B, L, 5) past weather-exog
            fut_weather (B, H, 5) future weather-exog (forecast/reanalysis)
            fut_exog    (B, H, 10) [sin_elev, clearsky, wind_cf, csi, cal×6]
            config      (B, D)   FD config vector (FD_CONFIG_FIELDS layout)
            ef_vec      (B, F)   per-fuel region EF vector (gCO2/kWh)
            hist_mask   (B, 1)   1 = history available (I_0), 0 = cold (I_cfg)

        Returns:
            cif (B, H) float, shares (B, H, F) float in [0, 1], rs_hat (B, H)
        """
        B, L = x_rs.shape
        H = self.horizon
        F = _N_FUEL
        if hist_mask is None:
            hist_mask = torch.ones(B, 1, device=x_rs.device, dtype=x_rs.dtype)
        hm = hist_mask  # (B, 1)

        # Gate raw history inputs (cold mode sees zeros).
        x_rs_g = x_rs * hm
        x_fuel_g = x_fuel * hm.unsqueeze(-1)
        x_weather_g = x_weather * hm.unsqueeze(-1)

        cfg = self.cfg_mlp(config)                       # (B, hidden)
        cfg_fuel = config[:, _CFG_FUEL0:_CFG_FUEL0 + F]  # (B, F)
        has_fuel = config[:, _CFG_HAS_FUEL:_CFG_HAS_FUEL + 1]  # (B, 1)

        # Per-sample generated weights for the dynamic heads (FD-2).  With
        # use_hypernet=False this is None and the plain shared heads run.
        gen = self.hypernet(config) if self.use_hypernet else None

        w168 = min(L, 168)
        # ------------------------------------------------------------------
        # 1. Solar: level × normalised clear-sky envelope × weather modulation
        # ------------------------------------------------------------------
        astro = fut_exog[:, :, 1]                        # (B, H) clearsky W/m^2
        astro_ref = astro.mean(dim=1, keepdim=True).clamp_min(10.0)
        astro_norm = astro / astro_ref                   # mean ≈ 1 over horizon

        hist_solar = x_fuel_g[:, -w168:, _IDX_SOLAR].mean(dim=1, keepdim=True)
        cfg_solar = cfg_fuel[:, _IDX_SOLAR:_IDX_SOLAR + 1]
        lvl_solar = torch.where(hm > 0.5, hist_solar, cfg_solar).clamp(0.0, 0.9)

        solar_mod = 1.0 + 0.4 * torch.tanh(
            _dyn_head("solar_mod", fut_weather, self.solar_mod, gen)).squeeze(-1)
        s_solar = lvl_solar * astro_norm * solar_mod     # (B, H)

        # ------------------------------------------------------------------
        # 2. Wind: level × normalised capacity factor × weather modulation
        # ------------------------------------------------------------------
        wcf_fut = fut_weather[:, :, 3]                   # (B, H)
        wcf_hist_ref = x_weather_g[:, -w168:, 3].mean(dim=1, keepdim=True)
        wcf_cfg_ref = config[:, _CFG_ANN_WINDCF:_CFG_ANN_WINDCF + 1].clamp_min(0.03)
        # Reference level: trailing week blended with the annual climatology
        # (a calm trailing week must not explode the normalisation).
        wcf_ref = torch.where(hm > 0.5, wcf_hist_ref, wcf_cfg_ref).clamp_min(0.02)
        wcf_ref = 0.7 * wcf_ref + 0.3 * wcf_cfg_ref
        # Bounded ratio: wind CF forecasts are noisy proxies, never 10x.
        wcf_norm = (wcf_fut / wcf_ref).clamp(0.2, 3.0)   # mean ~ 1 in history mode

        hist_wind = x_fuel_g[:, -w168:, _IDX_WIND].mean(dim=1, keepdim=True)
        cfg_wind = cfg_fuel[:, _IDX_WIND:_IDX_WIND + 1]
        lvl_wind = torch.where(hm > 0.5, hist_wind, cfg_wind).clamp(0.0, 0.95)

        wind_mod = 1.0 + 0.4 * torch.tanh(
            _dyn_head("wind_mod", fut_weather, self.wind_mod, gen)).squeeze(-1)
        s_wind = lvl_wind * wcf_norm * wind_mod          # (B, H)

        # ------------------------------------------------------------------
        # 3. Baseload: level (history or config) + support-masked correction
        # ------------------------------------------------------------------
        hist_base = x_fuel_g[:, -w168:, :][:, :, _BASELOAD_IDX].mean(dim=1)  # (B, 5)
        cfg_base = cfg_fuel[:, _BASELOAD_IDX]                                 # (B, 5)
        lvl_base = torch.where(hm > 0.5, hist_base, cfg_base)                 # (B, 5)

        # Config support mask: fuels absent from the annual mix stay absent.
        base_support = torch.sigmoid(20.0 * (cfg_base - 0.004))              # (B, 5)
        base_prior = self.base_prior(config)                                  # (B, 5)
        base_delta = _dyn_head("base_delta", fut_exog, self.base_delta, gen)        # (B, H, 5)
        supp = base_support.unsqueeze(1)                                      # (B, 1, 5)
        s_base = (lvl_base.unsqueeze(1)
                  + supp * base_prior.unsqueeze(1) + supp * base_delta)       # (B, H, 5)

        # ------------------------------------------------------------------
        # 4. Thermal residual: T = 1 - Σ(non-dispatchable); split anchored to
        #    the config thermal mix (log-space, zero-init == config prior)
        # ------------------------------------------------------------------
        s_nondisp = torch.cat([
            s_solar.unsqueeze(-1), s_wind.unsqueeze(-1),
            s_base.clamp_min(0.0)], dim=-1)               # (B, H, 7)
        nondisp_total = s_nondisp.sum(dim=-1)              # (B, H)
        thermal_total = (1.0 - nondisp_total).clamp(0.0, 1.0)

        cfg_thermal = cfg_fuel[:, _THERMAL_IDX]            # (B, 3)
        therm_anchor = torch.log(cfg_thermal + 0.02)       # config prior
        therm_support = torch.sigmoid(20.0 * (cfg_thermal - 0.004))
        therm_logits = ((therm_anchor
                         + therm_support * self.therm_cfg(config)).unsqueeze(1)
                        + therm_support.unsqueeze(1)
                        * _dyn_head("therm_dyn", fut_exog, self.therm_dyn, gen))
        therm_p = torch.softmax(therm_logits, dim=-1)      # (B, H, 3)
        s_thermal = thermal_total.unsqueeze(-1) * therm_p  # (B, H, 3)

        # ------------------------------------------------------------------
        # 5. Assemble + renormalise shares; physics layer with EF correction
        # ------------------------------------------------------------------
        shares = torch.zeros(B, H, F, device=x_rs.device, dtype=x_rs.dtype)
        shares[:, :, _IDX_SOLAR] = s_solar.clamp(0.0, 0.95)
        shares[:, :, _IDX_WIND] = s_wind.clamp(0.0, 0.95)
        shares[:, :, _BASELOAD_IDX] = s_base.clamp(0.0, 0.95)
        shares[:, :, _THERMAL_IDX] = s_thermal
        shares = shares / shares.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        ef_corr = 1.0 + 0.35 * torch.tanh(self.ef_corr(cfg))
        ef_eff = ef_vec * ef_corr                          # (B, F)
        cif_fuel = torch.einsum("bhf,bf->bh", shares, ef_eff)

        # ------------------------------------------------------------------
        # 6. Aggregate renewable-share head (AU path + auxiliary signal)
        # ------------------------------------------------------------------
        rs_hat = self._aggregate_head(x_rs_g, fut_exog, config, hm, gen=gen)
        ef_r = torch.zeros(B, device=x_rs.device, dtype=x_rs.dtype)
        ef_nr = config[:, _CFG_EF_NR] * 1000.0
        cif_agg = rs_hat * ef_r.unsqueeze(1) + (1.0 - rs_hat) * ef_nr.unsqueeze(1)

        # Regions without fuel telemetry take the aggregate path; fuel
        # regions take the per-fuel physics path.
        cif = has_fuel * cif_fuel + (1.0 - has_fuel) * cif_agg
        return cif, shares, rs_hat

    # ------------------------------------------------------------------
    def _aggregate_head(self, x_rs_g, fut_exog, config, hm, gen=None):
        """Aggregate renewable-share forecast (the proven flagship pattern,
        upgraded with future exog features and a cold-mode config level)."""
        B, L = x_rs_g.shape
        H = self.horizon
        # DLinear trend/seasonal decomposition (AvgPool smoothing, kernel 25)
        x3 = x_rs_g.unsqueeze(1)
        trend = torch.nn.functional.avg_pool1d(
            x3, kernel_size=25, stride=1, padding=12).squeeze(1)
        seasonal = x_rs_g - trend
        dlinear = self.rs_trend(trend) + self.rs_seasonal(seasonal)

        # Probability-space level: DLinear output in history mode, config
        # mean_rs in cold mode.  Anchored in LOGIT space so learned
        # corrections are additive offsets and the zero-init model returns
        # exactly the level prior (no cross-mode bias entanglement).
        cfg_lvl = config[:, _CFG_MEAN_RS:_CFG_MEAN_RS + 1]        # (B, 1)
        lvl = torch.where(hm > 0.5, dlinear.clamp(0.02, 0.98),
                          cfg_lvl.clamp(0.02, 0.98).expand_as(dlinear))
        anchor = torch.log(lvl / (1.0 - lvl))

        logits = (anchor + self.rs_cfg_bias(config)
                  + _dyn_head("rs_exog", fut_exog, self.rs_exog, gen).squeeze(-1))
        sig = torch.sigmoid(logits)
        # Persistence gate — history mode only.  In cold mode persistence is
        # unavailable (zeros); applying the gate there would drag the output
        # toward zero, so the cold path takes the sigmoid branch directly.
        persist = x_rs_g[:, -H:] if L >= H else x_rs_g
        recent_mean = x_rs_g[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x_rs_g[:, -48:].std(dim=1, keepdim=True)
        gate_in = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.rs_gate(gate_in))
        out = hm * (gate * persist + (1 - gate) * sig) + (1 - hm) * sig
        return out.clamp(0.0, 1.0)
