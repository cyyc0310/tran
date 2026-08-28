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
_IDX_IMPORTS = FUEL_INDEX["imports"]
_THERMAL_IDX = [FUEL_INDEX[f] for f in THERMAL_FUELS]
_BASELOAD_IDX = [FUEL_INDEX[f] for f in BASELOAD_FUELS]

# FD config layout (see data.fuel.FD_CONFIG_FIELDS)
_CFG_MEAN_RS = 0
_CFG_EF_NR = 1
_CFG_FUEL0 = 2                 # 10 fuel shares at [2:12]
_CFG_ANN_WINDCF = 12
_CFG_ANN_CSI = 13
_CFG_HAS_FUEL = 14
_CFG_WIND_SHARE = 2 + _IDX_WIND   # config slot of the annual wind share
_CFG_HYDRO_SHARE = 2 + FUEL_INDEX["hydro"]
# Weather-exog channel layout (see data.fuel.attach_fuel_and_exog): the
# trailing wind-regime channels enter here so the wind head can react to
# drought persistence and onset/exit ramps (extreme-weather attribution:
# CIF volatility peaks in the wind-share transition band, not in storms).
_WX_WIND_CF = 3
_WX_REGIME24 = 8
_WX_TEND6 = 9
# Deterministic structure router (roadmap #5, zero learned parameters).
# The actual threshold is an instance option; the recommended default is
# fuel-first (1.1) for fuel-telemetry domains after FD-17's weather fixes.
_WIND_ROUTE_TAU = 1.1


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

    def __init__(self, seq_len=336, horizon=24, n_weather=10, n_exog=17,
                 config_dim=16, hidden=32, use_hypernet=False,
                 ef_corr_bound=0.35, solar_mod_bound=0.4,
                 wind_route_tau=1.1, dynamic_residual=False,
                 dynamic_residual_bound=220.0):
        super().__init__()
        self.horizon = horizon
        self.n_weather = n_weather
        self.n_exog = n_exog
        self.use_hypernet = use_hypernet
        # Router threshold as an instance attribute.  After the FD-17
        # wind-unit/farm-weighting fixes, the full LORO sweep found that the
        # fuel path is the safer default for regions with fuel telemetry;
        # telemetry-free regions still take the aggregate path via
        # ``has_fuel``.  tau >= 1.1 means fuel-first, while 0.0 means
        # aggregate-first.  Keep this configurable for ablations.
        self.wind_route_tau = float(wind_route_tau)
        self.ef_corr_bound = ef_corr_bound
        self.solar_mod_bound = solar_mod_bound
        self.dynamic_residual = bool(dynamic_residual)
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
        # Shared temporal encoder for the future information set.  The old
        # heads consumed each future hour independently, which cannot see a
        # wind ramp or a multi-hour duck-curve transition.  The projection
        # heads below are zero-initialised, so this is an exact FD baseline
        # at step zero and becomes a low-capacity temporal residual during
        # training (no target labels are used at inference).
        future_in = n_weather + n_exog
        self.future_context = nn.Sequential(
            nn.Conv1d(future_in, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.solar_ctx = nn.Linear(16, 1)
        self.wind_ctx = nn.Linear(16, 1)
        self.base_ctx = nn.Linear(16, len(_BASELOAD_IDX))
        self.therm_ctx = nn.Linear(16, len(_THERMAL_IDX))
        self.rs_ctx = nn.Linear(16, 1)
        # Dynamic carbon-flow residual: static fuel EFs cannot represent
        # hourly fleet dispatch, imports, or plant-specific intensity.  This
        # bounded head learns the transferable weather/calendar component of
        # that error from source grids.  Zero initialization preserves the
        # original physics prior at step zero.
        self.cif_residual = nn.Sequential(
            nn.Linear(16 + n_exog + hidden, 32), nn.GELU(),
            nn.Linear(32, 1),
        )
        self.cif_residual_bound = float(dynamic_residual_bound)
        # --- learned bounded EF correction (multiplicative, per fuel);
        #     consumes the encoded config (hidden-dim), not the raw vector
        self.ef_corr = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, F),
        )
        # --- dedicated imports-EF pathway (FD-18): interconnector source
        #     mixes swing far beyond the shared ±35 % correction (French
        #     nuclear ≈ 50 vs Dutch gas ≈ 450 against a 250 canonical
        #     base).  Conditioned on config + the horizon's calendar
        #     state (hour/weekday cycles track neighbouring systems'
        #     export patterns).  Zero-init == canonical base; for a
        #     telemetry-free Chinese province the base value is the
        #     monthly flow-weighted import EF from public trading data.
        self.imp_ef = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.imp_ef_bound = 0.9
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
        # --- I_0 level anchor (history mode): the ZS+ branch-0 mechanism
        #     ("model shape + observed level") ported in-model.  The anchor
        #     level comes from the OBSERVED renewable-share stream via the
        #     2-fuel identity, so it is legal at I_0 (no CIF history); a
        #     gate on recent window statistics scales the correction.
        self.anchor_gate = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.constant_(self.anchor_gate[-1].bias, 1.5)  # start near-anchored
        # --- cold-mode anchor gate (FD-34): zero-init (pass-through at
        #     start); learns how strongly the monthly-config level should
        #     anchor the telemetry-free prediction
        self.cold_anchor_gate = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.cold_anchor_gate[-1].weight)
        nn.init.zeros_(self.cold_anchor_gate[-1].bias)
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
        for m in (self.solar_ctx, self.wind_ctx, self.base_ctx,
                  self.therm_ctx, self.rs_ctx):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.cif_residual[-1].weight)
        nn.init.zeros_(self.cif_residual[-1].bias)
        nn.init.zeros_(self.imp_ef[-1].weight)
        nn.init.zeros_(self.imp_ef[-1].bias)
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
            x_weather   (B, L, W) past weather-exog (W = 10: temp, swrad,
                                wind100, wind CF, csi, gust, MSL pressure,
                                demand z, wind regime 24 h, regime tendency)
            fut_weather (B, H, W) future weather-exog (forecast/reanalysis)
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

        # (B,H,W+E) -> (B,H,16), preserving future-hour order.
        future_input = torch.cat([fut_weather, fut_exog], dim=-1)
        future_ctx = self.future_context(
            future_input.transpose(1, 2)).transpose(1, 2)

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

        solar_signal = (_dyn_head("solar_mod", fut_weather, self.solar_mod, gen)
                        + self.solar_ctx(future_ctx))
        solar_mod = 1.0 + self.solar_mod_bound * torch.tanh(
            solar_signal).squeeze(-1)
        s_solar = lvl_solar * astro_norm * solar_mod     # (B, H)

        # ------------------------------------------------------------------
        # 2. Wind: level × normalised capacity factor × weather modulation
        # ------------------------------------------------------------------
        wcf_fut = fut_weather[:, :, _WX_WIND_CF]         # (B, H)
        wcf_hist_ref = x_weather_g[:, -w168:, _WX_WIND_CF].mean(dim=1, keepdim=True)
        wcf_cfg_ref = config[:, _CFG_ANN_WINDCF:_CFG_ANN_WINDCF + 1].clamp_min(0.03)
        # Reference level: trailing week blended with the annual climatology
        # (a calm trailing week must not explode the normalisation).
        wcf_ref = torch.where(hm > 0.5, wcf_hist_ref, wcf_cfg_ref).clamp_min(0.02)
        wcf_ref = 0.7 * wcf_ref + 0.3 * wcf_cfg_ref
        # Drought anchoring (FD-16): when the trailing-24 h weather regime
        # confirms a lull (regime/annual well below 1), pull the reference
        # toward the ANNUAL climatology — a week-long drought must not be
        # normalised away by an equally-calm trailing week.  The extreme-
        # weather attribution showed the wind-share transition band is
        # where CIF volatility (and model error) peaks.
        reg_ratio = None
        if fut_weather.shape[2] > _WX_REGIME24:
            reg_ratio = (fut_weather[:, :, _WX_REGIME24]
                         / wcf_cfg_ref).clamp(0.05, 2.0).mean(
                             dim=1, keepdim=True)
        if reg_ratio is not None:
            lull = torch.sigmoid(8.0 * (0.75 - reg_ratio))  # 1 -> lull
            wcf_ref = (1.0 - 0.6 * lull) * wcf_ref + 0.6 * lull * wcf_cfg_ref
        # Bounded ratio: wind CF forecasts are noisy proxies, never 10x.
        wcf_norm = (wcf_fut / wcf_ref).clamp(0.2, 3.0)   # mean ~ 1 in history mode

        hist_wind = x_fuel_g[:, -w168:, _IDX_WIND].mean(dim=1, keepdim=True)
        cfg_wind = cfg_fuel[:, _IDX_WIND:_IDX_WIND + 1]
        lvl_wind = torch.where(hm > 0.5, hist_wind, cfg_wind).clamp(0.0, 0.95)

        wind_signal = (_dyn_head("wind_mod", fut_weather, self.wind_mod, gen)
                       + self.wind_ctx(future_ctx))
        wind_mod = 1.0 + 0.4 * torch.tanh(wind_signal).squeeze(-1)
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
        base_delta = (_dyn_head("base_delta", fut_exog, self.base_delta, gen)
                      + self.base_ctx(future_ctx))                    # (B, H, 5)
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
                         * (_dyn_head("therm_dyn", fut_exog, self.therm_dyn, gen)
                            + self.therm_ctx(future_ctx)))
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

        ef_corr = 1.0 + self.ef_corr_bound * torch.tanh(self.ef_corr(cfg))
        ef_eff = ef_vec * ef_corr                          # (B, F)
        # Imports-EF pathway (FD-18): hour-of-day phase summary lets the
        # import emission factor track neighbouring systems' export
        # cycles; bounded ±0.9 around the canonical/flow-weighted base.
        cal_phase = fut_exog[:, :, 4:6].mean(dim=1)        # (B, 2)
        imp_lvl = torch.tanh(self.imp_ef(
            torch.cat([config, cal_phase], dim=1))).squeeze(-1)   # (B,)
        ef_eff = ef_eff.clone()
        ef_eff[:, _IDX_IMPORTS] = ef_vec[:, _IDX_IMPORTS] * (
            1.0 + self.imp_ef_bound * imp_lvl)
        cif_fuel = torch.einsum("bhf,bf->bh", shares, ef_eff)
        # Source-trained, bounded dynamic carbon-flow correction.  The
        # config embedding provides domain structure; future context and
        # exogenous variables provide the deployment-available state.
        cfg_seq = cfg.unsqueeze(1).expand(-1, H, -1)
        residual_in = torch.cat([future_ctx, fut_exog, cfg_seq], dim=-1)
        cif_residual = self.cif_residual_bound * torch.tanh(
            self.cif_residual(residual_in)).squeeze(-1)
        if not self.dynamic_residual:
            cif_residual = torch.zeros_like(cif_residual)
        cif_fuel = cif_fuel + cif_residual

        # ------------------------------------------------------------------
        # 6. Aggregate renewable-share head (AU path + auxiliary signal)
        # ------------------------------------------------------------------
        rs_hat = self._aggregate_head(x_rs_g, fut_exog, config, hm, gen=gen,
                                      future_ctx=future_ctx)
        ef_r = torch.zeros(B, device=x_rs.device, dtype=x_rs.dtype)
        ef_nr = config[:, _CFG_EF_NR] * 1000.0
        cif_agg = rs_hat * ef_r.unsqueeze(1) + (1.0 - rs_hat) * ef_nr.unsqueeze(1)

        # Deterministic structure router: fuel telemetry enables the
        # decomposition path, but wind-heavy configs (>= _WIND_ROUTE_TAU)
        # fall back to the aggregate path whose shape rides on the
        # observed rs stream instead of weather representativeness.
        # Hydro-dominant grids (>= 0.5) also route aggregate (FD-19):
        # dispatchable hydro follows the load curve intraday, but the
        # baseload head models it as a slow level — the rs telemetry /
        # config+calendar aggregate path carries that shape instead
        # (empirically BPAT, 71 % hydro: I_cfg 46.7 -> 15.2).
        wind_cfg = config[:, _CFG_WIND_SHARE:_CFG_WIND_SHARE + 1]
        hydro_cfg = config[:, _CFG_HYDRO_SHARE:_CFG_HYDRO_SHARE + 1]
        route_fuel = torch.sigmoid(
            20.0 * (self.wind_route_tau - wind_cfg)) \
            * torch.sigmoid(30.0 * (0.5 - hydro_cfg)) * has_fuel
        cif = route_fuel * cif_fuel + (1.0 - route_fuel) * cif_agg

        # ------------------------------------------------------------------
        # 7. Level anchor: history mode anchors to the OBSERVED rs window
        #    (I_0, ZS+ branch-0 mechanism).  Cold mode (I_cfg) anchors to
        #    the CONFIG mean_rs slot (FD-34) — with the monthly interface
        #    that slot carries the target's lagged PUBLISHED monthly
        #    renewable share, the exact public-statistics substitute for
        #    the telemetry anchor (green/orange gap decomposition: the
        #    level component is +18 on CISO/NSW1-class regions).
        # ------------------------------------------------------------------
        cif = (cif + hm * self._anchor_correction(
            x_rs_g, config, cif, ef_nr)
               + (1.0 - hm) * self._cold_anchor(
                   config, cif, ef_nr)).clamp_min(0.0)

        return cif, shares, rs_hat

    def _cold_anchor(self, config, cif, ef_nr):
        """Cold-mode level anchor from the config mean_rs slot (FD-34).

        Zero-init gate -> a fresh model passes through unchanged; with the
        monthly interface config[0] is the lagged published monthly mean
        renewable share, so the anchor level (1 - mean_rs) * ef_nr tracks
        the target's seasonal level without any telemetry.
        """
        cfg_mean_rs = config[:, 0:1].clamp(0.0, 1.0)
        anchor_level = (1.0 - cfg_mean_rs) * ef_nr.unsqueeze(1)
        gate = torch.sigmoid(self.cold_anchor_gate(
            torch.cat([config], dim=1)))
        return gate * (anchor_level - cif.mean(dim=1, keepdim=True))

    def _anchor_correction(self, x_rs_g, config, cif, ef_nr):
        """Gated additive level correction from the observed rs window.

        Mechanism = TransCIF-ZS+ branch 0: model contributes the SHAPE,
        the recent observed renewable share contributes the LEVEL
        (``(1 - rs_anchor) * ef_nr`` in gCO2/kWh).  A gate on recent
        window statistics (config, recent mean, recent std) modulates the
        correction strength, initialised near-anchored (sigmoid bias 1.5)
        so training starts from the empirically strongest configuration
        and can relax it per regime.
        """
        recent_mean = x_rs_g[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x_rs_g[:, -48:].std(dim=1, keepdim=True)
        anchor_level = (1.0 - recent_mean.clamp(0.0, 1.0)) * ef_nr.unsqueeze(1)
        gate_in = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.anchor_gate(gate_in))
        return gate * (anchor_level - cif.mean(dim=1, keepdim=True))

    # ------------------------------------------------------------------
    def _aggregate_head(self, x_rs_g, fut_exog, config, hm, gen=None,
                        future_ctx=None):
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

        rs_signal = _dyn_head("rs_exog", fut_exog, self.rs_exog, gen)
        if future_ctx is not None:
            rs_signal = rs_signal + self.rs_ctx(future_ctx)
        logits = (anchor + self.rs_cfg_bias(config)
                  + rs_signal.squeeze(-1))
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
