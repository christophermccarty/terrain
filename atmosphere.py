"""Simple 3-cell-per-hemisphere wind model for equirectangular maps.

Hadley, Ferrel, and Polar cells are approximated by prescribing zonal (u)
and meridional (v) surface winds by latitude, with Coriolis turning that
creates easterlies and westerlies in the appropriate bands.

This is intentionally lightweight for interactive use; it returns a dense
vector field and a pre-rendered arrow RGB overlay for display.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np
from temperature import temperature_kelvin_for_lat


def _latitudes_h(height: int) -> np.ndarray:
    # Row-centered latitudes θ ∈ [π/2, -π/2] (north to south)
    return (0.5 - (np.arange(int(height), dtype=np.float32) + 0.5) / float(height)) * np.pi


def _coarse_shape(H: int, W: int, block_size: int) -> tuple[int, int]:
    bs = max(1, int(block_size))
    Hc = max(1, (H + bs - 1) // bs)
    Wc = max(1, (W + bs - 1) // bs)
    return Hc, Wc


def _upsample_repeat(field: np.ndarray, H: int, W: int, block_size: int) -> np.ndarray:
    bs = max(1, int(block_size))
    up = np.repeat(np.repeat(field, bs, axis=0), bs, axis=1)
    return up[:H, :W]


def _upsample_bilinear(field: np.ndarray, H: int, W: int, block_size: int) -> np.ndarray:
    """Upsample using bilinear interpolation to eliminate blocky artifacts."""
    bs = max(1, int(block_size))
    Hc, Wc = field.shape
    if bs == 1:
        return field[:H, :W]

    y0, y1, wy, x0, x1, wx = _bilinear_plan(int(H), int(W), int(Hc), int(Wc))
    f0 = field[y0, :].astype(np.float32, copy=False)
    f1 = field[y1, :].astype(np.float32, copy=False)
    top = f0[:, x0] * (1.0 - wx)[None, :] + f0[:, x1] * wx[None, :]
    bot = f1[:, x0] * (1.0 - wx)[None, :] + f1[:, x1] * wx[None, :]
    return (top * (1.0 - wy)[:, None] + bot * wy[:, None]).astype(np.float32)


@lru_cache(maxsize=64)
def _bilinear_plan(H: int, W: int, Hc: int, Wc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 1D sampling plan; avoids per-call 2D meshgrid/indices allocations.
    y = np.linspace(0, Hc - 1, int(H), dtype=np.float32)
    x = np.linspace(0, Wc - 1, int(W), dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    y0 = np.clip(y0, 0, Hc - 1)
    x0 = np.clip(x0, 0, Wc - 1)
    y1 = np.clip(y0 + 1, 0, Hc - 1)
    x1 = np.clip(x0 + 1, 0, Wc - 1)
    return y0, y1, wy, x0, x1, wx


def _upsample_bilinear_many(fields: dict[str, np.ndarray], H: int, W: int, block_size: int) -> dict[str, np.ndarray]:
    """Upsample multiple (Hc,Wc) fields sharing the same sampling plan."""
    if not fields:
        return {}
    bs = max(1, int(block_size))
    first = next(iter(fields.values()))
    Hc, Wc = first.shape
    if bs == 1:
        return {k: v[:H, :W] for k, v in fields.items()}

    keys = list(fields.keys())
    stack = np.stack([fields[k].astype(np.float32, copy=False) for k in keys], axis=0)
    y0, y1, wy, x0, x1, wx = _bilinear_plan(int(H), int(W), int(Hc), int(Wc))
    f0 = stack[:, y0, :]
    f1 = stack[:, y1, :]
    top = f0[:, :, x0] * (1.0 - wx)[None, None, :] + f0[:, :, x1] * wx[None, None, :]
    bot = f1[:, :, x0] * (1.0 - wx)[None, None, :] + f1[:, :, x1] * wx[None, None, :]
    out = (top * (1.0 - wy)[None, :, None] + bot * wy[None, :, None]).astype(np.float32)
    return {k: out[i] for i, k in enumerate(keys)}


def _majority_filter(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(np.int8)
    for _ in range(max(1, iterations)):
        pad = np.pad(out, 1, mode="edge")
        neigh = (
            pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
            + pad[1:-1, 0:-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
            + pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
        )
        out = (neigh >= 5).astype(np.int8)
    return out.astype(bool)


def _derive_land_sea_masks(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    elev = elevation.astype(np.float32)
    # If elevation contains a meaningful ocean mask (common in loaded DEM workflow),
    # treat near-zero as ocean instead of using a median split.
    ocean_eps = 1e-6
    zeros_frac = float(np.mean(elev <= ocean_eps))
    if zeros_frac > 0.02:
        sea = elev <= ocean_eps
        land = ~sea
    else:
        thresh = float(np.median(elev))
        land = elev > thresh
    land = _majority_filter(land, iterations=2)
    sea = ~land
    return land, sea


def _laplacian(field: np.ndarray) -> np.ndarray:
    # Periodic in longitude (axis=1), clamped at poles (axis=0).
    pad_y = np.pad(field, ((1, 1), (0, 0)), mode="edge")
    c = pad_y[1:-1, :]
    n = pad_y[0:-2, :]
    s = pad_y[2:, :]
    e = np.roll(c, -1, axis=1)
    w = np.roll(c, 1, axis=1)
    return n + s + e + w - 4.0 * c


def _ddx_periodic(field: np.ndarray) -> np.ndarray:
    """Central difference in x with periodic wrap (axis=1). Returns derivative per grid index."""
    return 0.5 * (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1))


def _advect_scalar(
    field: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    u_scale: np.ndarray,
    v_scale: np.ndarray,
) -> np.ndarray:
    east = np.roll(field, -1, axis=1)
    west = np.roll(field, 1, axis=1)
    north = np.roll(field, -1, axis=0)
    south = np.roll(field, 1, axis=0)
    adv_x = field + u_scale * (np.where(u >= 0, west, east) - field)
    adv_xy = adv_x + v_scale * (np.where(v >= 0, south, north) - adv_x)
    return adv_xy


def _streamfunction_from_vorticity(omega: np.ndarray) -> np.ndarray:
    H, W = omega.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(H)
    kx = 2.0 * np.pi * np.fft.rfftfreq(W)
    K2 = ky[:, None] ** 2 + kx[None, :] ** 2
    omega_hat = np.fft.rfft2(omega)
    psi_hat = np.zeros_like(omega_hat)
    mask = K2 > 1e-9
    psi_hat[mask] = -omega_hat[mask] / K2[mask]
    psi_hat[0, 0] = 0.0
    psi = np.fft.irfft2(psi_hat, s=omega.shape)
    return psi.astype(np.float32)


def evolve_wind(
    u: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray | None,
    elevation: np.ndarray,
    dt_days: float = 1.0,
    damping: float = 0.25,
    pgf_temp_scale: float = 450.0,
    pgf_terrain_scale: float = 900.0,
    drag_base: float = 2.0e-7,
    drag_elev_scale: float = 6.0e-7,
    vmax_clip: float = 150.0,
    baroclinic_jet_amp: float = 0.0,
    baroclinic_mix: float = 0.0,
    cell_relax_days: float = 0.0,
    time_days: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve wind field using simplified primitive momentum equations.

    Equations:
    du/dt = - (u*du/dx + v*du/dy) + f*v - (1/rho)*dp/dx + F_x
    dv/dt = - (u*dv/dx + v*dv/dy) - f*u - (1/rho)*dp/dy + F_y

    Physics included:
    - Advection (self-transport)
    - Coriolis force (rotation)
    - Pressure Gradient Force (thermal + dynamic)
    - Friction (surface drag)
    - Jet stream dynamics (thermal wind balance)
    """
    H, W = u.shape
    dt_total = dt_days * 86400.0  # seconds
    
    # Sub-stepping for stability (CFL-like). Full-res winds have smaller dx than coarse winds,
    # so a fixed 1-hour substep can violate CFL and force the vmax clamp.
    # Use a conservative estimate based on grid spacing and vmax_clip.
    # (dx→0 near poles; use a low-percentile to avoid dt→0.)
    # Target CFL ~ 0.4
    R_earth = 6.371e6
    lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
    lat_2d = np.repeat(lat[:, None], W, axis=1)
    dx = R_earth * (2 * np.pi / W) * np.cos(lat_2d)
    dy = R_earth * (np.pi / H)
    dx_eff = dx[np.isfinite(dx)]
    dx_min = float(np.nanpercentile(dx_eff, 5.0)) if dx_eff.size else float(dy)
    dx_min = max(1.0e3, dx_min)
    dy_min = float(dy)
    dt_cfl = 0.4 * min(dx_min, dy_min) / max(1.0, float(vmax_clip))
    substep_sec = float(np.clip(dt_cfl, 120.0, 3600.0))
    n_steps = max(1, int(np.ceil(dt_total / substep_sec)))
    dt_sub = dt_total / n_steps

    # Grid parameters
    # Coriolis parameter f = 2*Omega*sin(lat)
    Omega = 7.2921e-5
    f = 2.0 * Omega * np.sin(lat_2d)
    abs_lat_deg_2d = np.abs(np.rad2deg(lat_2d)).astype(np.float32)
    # Equatorial damping: in a single-layer model, PGF can over-accelerate winds where f≈0.
    # Boost drag within ~±12° to recover calmer doldrums and prevent equatorial jets.
    eq_window = np.exp(-((abs_lat_deg_2d / 12.0) ** 2)).astype(np.float32)
    
    # Gradients dx, dy
    # (dx,dy already computed above)
    
    rho = 1.225 # kg/m3
    
    u_curr, v_curr = u.copy(), v.copy()
    
    # Pre-calculate PGF (constant over the day)
    if pressure is None:
        # P ~ P0 * exp(-z/H) * (T0/T)^g/R... simplified:
        # NOTE: Keep this in Pa (not hPa). Scale is intentionally "synoptic-ish":
        # order ~5-10 hPa swings across large temperature gradients.
        p_anom = -float(pgf_temp_scale) * ((temperature - 273.15) / 30.0)  # Pa anomaly
        if elevation is not None:
             # Terrain effect: flow around obstacles, high pressure wedge
             p_anom += float(pgf_terrain_scale) * elevation 
    else:
        p_anom = pressure

    # Add weak, time-varying mid-latitude planetary waves to break perfectly zonal bands.
    # This produces storm-track-like meanders instead of uniform stripes.
    if time_days is not None:
        lon = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)
        abs_deg_1d = np.rad2deg(np.abs(lat_2d[:, 0])).astype(np.float32)  # (H,)
        storm_w = np.exp(-((abs_deg_1d - 45.0) / 18.0) ** 2).astype(np.float32)  # (H,)
        t = float(time_days)
        wave = np.zeros((H, W), dtype=np.float32)
        # hPa-scale perturbations -> Pa
        for k, per, ph, amp_hpa in ((3.0, 6.0, 0.3, 1.2), (5.0, 9.0, 1.1, 0.9), (7.0, 14.0, -0.7, 0.6)):
            wave += (amp_hpa * 100.0) * np.cos(k * lon[None, :] + (2.0 * np.pi * t / per) + ph).astype(np.float32)
        p_anom = p_anom + storm_w[:, None] * wave
        
    dp_dx = _ddx_periodic(p_anom) / (dx + 1e-3)
    # Axis 0 is north→south (index increases southward), so physical northward gradient is negated.
    dp_dy = -np.gradient(p_anom, axis=0) / dy
    
    pgf_u = -1.0/rho * dp_dx
    pgf_v = -1.0/rho * dp_dy

    # --- Baroclinic / eddy-driven mid-lat westerly tendency (parameterized) ---
    # In the real atmosphere, mid-lat westerlies are maintained by baroclinic eddies and
    # their momentum flux convergence. We approximate the net effect by mixing a
    # thermal-wind-like westerly target down toward the surface, based on |dT/dy|.
    b_amp = float(baroclinic_jet_amp)
    b_mix = float(baroclinic_mix)
    if b_amp != 0.0 and b_mix > 0.0:
        # Use zonal-mean temperature gradient so the tendency acts on the zonal-mean jet,
        # matching the climatological effect of baroclinic eddies.
        abs_deg_1d = np.rad2deg(np.abs(lat_2d[:, 0])).astype(np.float32)  # (H,)
        w_mid_1d = np.exp(-((abs_deg_1d - 45.0) / 12.0) ** 2).astype(np.float32)  # (H,)
        ztemp = np.mean(temperature.astype(np.float32), axis=1, keepdims=True)  # (H,1)
        dT_dy = -np.gradient(ztemp, axis=0) / dy  # (H,1) physical K/m; negate for north→south axis
        u_jet = (b_amp * w_mid_1d[:, None] * np.abs(dT_dy)).astype(np.float32)  # (H,1)
        # Safety clamp: keep the parameterization from generating unrealistic surface jets.
        u_jet = np.clip(u_jet, 0.0, 70.0).astype(np.float32)

    # --- 3-cell surface tendency (Hadley/Ferrel/Polar) ---
    # A single-layer model won't spontaneously generate the full overturning circulation.
    # This optional, weak relaxation nudges zonal-mean (u,v) toward an Earth-like 3-cell
    # surface signature: trades (easterly), mid-lat westerlies, polar easterlies; plus
    # equatorward/poleward v bands by hemisphere.
    tau_cell = float(cell_relax_days)
    if tau_cell > 0.0:
        abs_deg_1d = np.rad2deg(np.abs(lat_2d[:, 0])).astype(np.float32)  # (H,)
        sign_lat = np.sign(lat_2d[:, 0]).astype(np.float32)  # +N, -S
        # Broaden the windows + reduce amplitudes to avoid razor-thin zonal bands.
        w_trade = np.exp(-((abs_deg_1d - 15.0) / 12.0) ** 2).astype(np.float32)
        w_mid = np.exp(-((abs_deg_1d - 45.0) / 18.0) ** 2).astype(np.float32)
        w_polar = np.exp(-((abs_deg_1d - 75.0) / 14.0) ** 2).astype(np.float32)
        # Keep polar surface easterlies: ensure the polar term dominates at high latitudes.
        u_target = (-2.0 * w_trade + 3.0 * w_mid - 5.0 * w_polar).astype(np.float32)  # m/s
        # v_target: Hadley (equatorward), Ferrel (poleward), Polar (equatorward), by hemisphere.
        v_target = (-1.6 * w_trade + 5.0 * w_mid - 3.0 * w_polar).astype(np.float32) * sign_lat  # m/s
        # Remove the equator sign ambiguity (sign(0)=0) so the equator stays calm.
        v_target = np.where(np.abs(lat_2d[:, 0]) < np.deg2rad(2.0), 0.0, v_target).astype(np.float32)
        k_cell = 1.0 / (tau_cell * 86400.0)
    
    for _ in range(n_steps):
        # Advection
        u_scale = np.clip(np.abs(u_curr) * dt_sub / (dx + 1e-3), 0, 0.5)
        v_scale = np.clip(np.abs(v_curr) * dt_sub / dy, 0, 0.5)
        u_adv = _advect_scalar(u_curr, u_curr, v_curr, u_scale, v_scale)
        v_adv = _advect_scalar(v_curr, u_curr, v_curr, u_scale, v_scale)
        
        # Friction (quadratic surface drag). Keep small; dt_sub is in seconds.
        drag = float(drag_base)
        if elevation is not None:
            drag += float(drag_elev_scale) * elevation
        drag = drag + (2.0e-6 * eq_window)
        friction_u = -drag * u_curr * np.abs(u_curr)
        friction_v = -drag * v_curr * np.abs(v_curr)
        
        # Integration
        du = (pgf_u + f * v_curr + friction_u) * dt_sub
        dv = (pgf_v - f * u_curr + friction_v) * dt_sub

        # Mix toward baroclinic jet target (eddy momentum flux convergence proxy)
        if b_amp != 0.0 and b_mix > 0.0:
            # relaxation rate (1/s)
            k = 1.0 / (b_mix * 86400.0)
            u_zm = np.mean(u_curr, axis=1, keepdims=True)  # (H,1)
            du = du + (u_jet - u_zm) * k * dt_sub

        u_curr = u_adv + du * damping
        v_curr = v_adv + dv * damping

        # Relax zonal-mean toward 3-cell surface targets (apply directly so it isn't
        # weakened by the global `damping` factor above).
        if tau_cell > 0.0:
            a = float(np.clip(dt_sub * k_cell, 0.0, 1.0))
            u_zm = np.mean(u_curr, axis=1, keepdims=True)  # (H,1)
            v_zm = np.mean(v_curr, axis=1, keepdims=True)  # (H,1)
            # Pull u toward the target in mid-lats + polar regions to avoid razor-thin, overly-fast jets.
            u_t = np.clip(u_target, -10.0, 10.0).astype(np.float32)
            a_u_row = np.clip(a * (1.0 + 20.0 * w_mid[:, None] + 30.0 * w_polar[:, None]), 0.0, 1.0).astype(np.float32)
            u_curr = u_curr + (u_t[:, None] - u_zm) * a_u_row
            # Relax v more strongly in mid-lats, but keep it modest so bands can meander.
            # (Ferrel surface flow is otherwise easily flipped by Coriolis coupling from strong u.)
            a_v_row = np.clip(a * (3.0 + 90.0 * w_mid[:, None] + 150.0 * w_polar[:, None]), 0.0, 1.0).astype(np.float32)
            v_curr = v_curr + (v_target[:, None] - v_zm) * a_v_row
        
        # Soft clamp to prevent explosion
        total_v = np.hypot(u_curr, v_curr)
        vmax = float(vmax_clip)
        mask_high = total_v > vmax
        scale = vmax / (total_v + 1e-6)
        u_curr[mask_high] *= scale[mask_high]
        v_curr[mask_high] *= scale[mask_high]
    
    return u_curr.astype(np.float32), v_curr.astype(np.float32)


def generate_wind_field(
    height: int,
    width: int,
    *,
    day_of_year: int = 80,
    block_size: int = 3,
    upsample: str = "repeat",
    temperature: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
    terrain_influence: float = 1.0,
    weather_amp: float = 1.0,
    zonal_pressure: float = 0.85,
    terrain_pressure_amp: float = 1.0,
    terrain_flow_amp: float = 1.0,
    time_days: float | None = None,
    debug_log: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) near-surface winds derived from pressure gradients.

    Build surface pressure from temperature (with land-sea contrast and seasonal
    variation), add terrain and weather-system perturbations, then derive winds
    from pressure gradients via geostrophic balance. A streamfunction solver
    ensures divergence-free flow while preserving realistic meridional components.
    """

    H = int(height)
    W = int(width)
    Hc, Wc = _coarse_shape(H, W, block_size)
    lat = _latitudes_h(Hc)
    lon = np.linspace(-np.pi, np.pi, Wc, endpoint=False)
    abs_deg = np.rad2deg(np.abs(lat))

    # Prepare terrain and land/sea masks
    elev_c: np.ndarray
    land_mask: np.ndarray
    sea_mask: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    
    if elevation is not None:
        elev_pad = np.pad(
            elevation.astype(np.float32),
            ((0, Hc * block_size - H), (0, Wc * block_size - W)),
            mode="edge",
        )
        elev_c = elev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
        land_mask, sea_mask = _derive_land_sea_masks(elev_c)
        gx, gy = np.gradient(elev_c)
    else:
        elev_c = np.zeros((Hc, Wc), dtype=np.float32)
        land_mask = np.zeros((Hc, Wc), dtype=bool)
        sea_mask = np.ones((Hc, Wc), dtype=bool)
        gx = np.zeros((Hc, Wc), dtype=np.float32)
        gy = np.zeros((Hc, Wc), dtype=np.float32)

    # Temperature field:
    # - If provided, use it (it should come from the simulation state).
    # - Otherwise fall back to a lightweight climatology.
    if temperature is not None:
        temp = temperature.astype(np.float32)
        if temp.shape == (H, W):
            temp_pad = np.pad(
                temp,
                ((0, Hc * block_size - H), (0, Wc * block_size - W)),
                mode="edge",
            )
            T = temp_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
        elif temp.shape == (Hc, Wc):
            T = temp
        else:
            raise ValueError(f"temperature must be shape {(H, W)} or {(Hc, Wc)}; got {temp.shape}")
        # Mild smoothing only (keep gradients that actually drive winds).
        T = np.clip(T + 0.05 * _laplacian(T), 200.0, 330.0)
        season_phase = 2.0 * np.pi * (day_of_year - 80) / 365.2422
        land_f = land_mask.astype(np.float32)
    else:
        # Build 2D temperature field with land-sea contrast and longitudinal variation
        T_lat = temperature_kelvin_for_lat(lat, day_of_year=day_of_year).astype(np.float32)
        T = np.repeat(T_lat[:, None], Wc, axis=1).astype(np.float32)
        
        # Land-sea temperature contrast (land warmer in summer, cooler in winter)
        land_f = land_mask.astype(np.float32)
        season_phase = 2.0 * np.pi * (day_of_year - 80) / 365.2422
        seasonal_contrast = 8.0 * np.sin(season_phase) * np.cos(lat[:, None])  # NH summer = positive
        T += seasonal_contrast * land_f
        
        # Coastal gradients (sharp temperature transitions)
        coastal_grad = _laplacian(land_f)
        T += 3.5 * coastal_grad * (T / 280.0)
        
        # Add longitudinal temperature waves (continentality, monsoon drivers)
        T_wave = 6.0 * np.sin(lon[None, :] * 2.5 + season_phase) * land_f
        T_wave += 3.5 * np.sin(lon[None, :] * 4.0 - 0.8) * land_f
        T_wave *= np.exp(-((abs_deg[:, None] - 35.0) / 25.0) ** 2)  # Peak at mid-latitudes
        T += T_wave
        
        # Smooth temperature to avoid numerical instabilities
        T = np.clip(T + 0.15 * _laplacian(T), 200.0, 320.0)
    
    # Convert temperature to surface pressure (ideal gas law approximation)
    # For sim-driven winds we keep it mostly zonal to avoid stationary continent-scale blobs.
    T_ref = 273.15
    zp = float(np.clip(zonal_pressure, 0.0, 1.0))
    if temperature is not None and zp > 0.0:
        T_zonal = np.mean(T, axis=1, keepdims=True)
        T_used = zp * T_zonal + (1.0 - zp) * T
    else:
        T_used = T
    # Warmer = lower pressure (thermal low), colder = higher pressure (thermal high)
    p_thermal = 1013.25 * (T_ref / (T_used + 1e-6)) ** 2.2  # hPa
    
    # Add terrain pressure anomalies (mountains create blocking highs)
    tp = float(np.clip(terrain_pressure_amp, 0.0, 1.0))
    p_terrain = 25.0 * tp * terrain_influence * np.clip(elev_c, 0.0, 1.0)
    
    # Optional: inject weak synoptic-scale perturbations.
    wamp = float(np.clip(weather_amp, 0.0, 1.0))
    if wamp > 0.0:
        # If simulation time is provided, use traveling-wave perturbations (moving eddies)
        # instead of stationary, day-seeded blobs.
        t_days = float(time_days) if time_days is not None else float(day_of_year)
        if temperature is not None:
            # Mid-latitude storm-track window
            storm_window = np.exp(-((abs_deg[:, None] - 45.0) / 18.0) ** 2).astype(np.float32)
            equ_window = np.exp(-((abs_deg[:, None]) / 12.0) ** 2).astype(np.float32)
            # A small set of deterministic planetary waves
            ks = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float32)
            periods = np.array([6.0, 9.0, 14.0, 20.0], dtype=np.float32)  # days
            phases = np.array([0.3, 1.1, -0.7, 2.2], dtype=np.float32)
            amps = np.array([6.0, 4.5, 3.0, 2.5], dtype=np.float32) * wamp  # hPa
            wave = np.zeros((Hc, Wc), dtype=np.float32)
            for k, per, ph, a in zip(ks, periods, phases, amps):
                wave += a * np.cos(k * lon[None, :] + (2.0 * np.pi * t_days / per) + ph)
            p_thermal += storm_window * wave

            # Add a weak equatorial traveling wave (Walker / MJO-ish), to avoid overly static tropics.
            p_thermal += equ_window * (2.0 * wamp) * np.cos(2.0 * lon[None, :] - (2.0 * np.pi * t_days / 7.0) + 0.4)
        else:
            # Fallback (static view): small stationary systems
            rng = np.random.default_rng(int(day_of_year) + 9001)
            n_systems = 6
            for _ in range(n_systems):
                sys_lon = rng.uniform(-np.pi, np.pi)
                sys_lat = rng.uniform(-0.6 * np.pi, 0.6 * np.pi)
                sys_strength = rng.uniform(-12.0, 12.0) * wamp
                sys_scale = rng.uniform(0.18, 0.40)
                
                dist_lon = np.abs(lon[None, :] - sys_lon)
                dist_lon = np.minimum(dist_lon, 2.0 * np.pi - dist_lon)
                dist_lat = np.abs(lat[:, None] - sys_lat)
                dist = np.sqrt(dist_lon ** 2 + dist_lat ** 2)
                
                p_system = sys_strength * np.exp(-(dist / sys_scale) ** 2)
                p_thermal += p_system
    
    # Total pressure field
    pressure = p_thermal + p_terrain
    pressure = pressure + 0.2 * _laplacian(pressure)  # Smooth
    
    # Derive geostrophic winds from pressure gradients
    # In geostrophic balance: u ∝ -∂p/∂y, v ∝ ∂p/∂x (Northern Hemisphere)
    # Scale by Coriolis parameter f = 2Ω sin(φ)
    f_coriolis = 2.0 * 7.2921e-5 * np.sin(lat)  # Earth's rotation rate
    # Avoid division by zero at equator: enforce minimum magnitude while preserving sign
    f_min = 3e-5  # Increased from 1e-5 for stronger damping near equator
    mask_pos = f_coriolis >= 0
    f_coriolis = np.where(mask_pos, np.maximum(f_coriolis, f_min), np.minimum(f_coriolis, -f_min))
    
    # Axis 0 is north→south (index increases southward), so physical northward gradient is negated.
    # Use proper metric terms (meters), consistent with `evolve_wind`.
    R_earth = 6.371e6
    lat_2d = np.repeat(lat[:, None], Wc, axis=1)
    dx = R_earth * (2 * np.pi / Wc) * np.cos(lat_2d)
    dy = R_earth * (np.pi / Hc)
    p_pa = (pressure * 100.0).astype(np.float32)  # hPa -> Pa
    dp_dy = -np.gradient(p_pa, axis=0) / dy
    dp_dx = _ddx_periodic(p_pa) / (dx + 1e-3)
    
    # Geostrophic wind (m/s)
    rho = 1.2  # kg/m³ air density
    u_geo = -(1.0 / (rho * f_coriolis[:, None])) * dp_dy
    v_geo = (1.0 / (rho * f_coriolis[:, None])) * dp_dx
    
    # Add tropical wind model (trade winds/Walker circulation)
    # Geostrophic approximation breaks down near equator; use direct tropical circulation
    abs_lat = np.abs(lat)
    tropical_mask = abs_lat < np.deg2rad(25.0)  # ±25° tropical zone
    
    # Tropical winds (Hadley-cell-like):
    # - Doldrums near the equator (weak surface winds)
    # - Trades peak off-equator (~10-20°) and converge toward ITCZ.
    lat0 = np.deg2rad(25.0)
    absn = np.clip(abs_lat / lat0, 0.0, 1.0)
    trade_profile = np.sin(np.pi * absn)  # 0 at equator and 25°, peak near 12.5°
    u_tropical = -(2.8 * trade_profile[:, None]) * (1.0 + 0.12 * np.sin(lon[None, :] * 1.4))  # easterlies
    v_tropical = -0.6 * np.tanh(lat[:, None] / np.deg2rad(10.0)) * (1.0 - absn[:, None])  # toward equator
    
    # Blend: tropical model in tropics, geostrophic elsewhere
    # Tropical zones use primarily tropical model with small geostrophic component
    geo_scale_mid = 0.08
    geo_scale_trop = 0.02
    u_geo = np.where(tropical_mask[:, None], 
                     u_tropical + geo_scale_trop * u_geo,  # Small geostrophic contribution
                     geo_scale_mid * u_geo)  # Mid/high latitudes use geostrophic
    v_geo = np.where(tropical_mask[:, None],
                     v_tropical + geo_scale_trop * v_geo,
                     geo_scale_mid * v_geo)
    
    # Latitude-dependent clipping for realistic wind speeds
    tropical_limit_u = 12.0  # Weaker tropical winds
    midlat_limit_u = 22.0    # Stronger mid-latitude storm tracks
    tropical_limit_v = 8.0
    midlat_limit_v = 15.0
    
    # Scale limits linearly from equator (0) to 60° (1); equator keeps tropical caps
    lat_factor = np.clip(np.abs(lat) / np.deg2rad(60.0), 0.0, 1.0)
    u_limit = tropical_limit_u + (midlat_limit_u - tropical_limit_u) * lat_factor
    v_limit = tropical_limit_v + (midlat_limit_v - tropical_limit_v) * lat_factor
    
    u_geo = np.clip(u_geo, -u_limit[:, None], u_limit[:, None])
    v_geo = np.clip(v_geo, -v_limit[:, None], v_limit[:, None])
    
    # Add ageostrophic component (cross-isobar flow toward low pressure)
    ageo_frac = 0.03
    u_ageo = -ageo_frac * dp_dx / (np.abs(dp_dx) + 1e-3) * np.abs(u_geo) * 0.10
    v_ageo = -ageo_frac * dp_dy / (np.abs(dp_dy) + 1e-3) * np.abs(v_geo) * 0.10
    
    uc = u_geo + u_ageo
    vc = v_geo + v_ageo

    # --- Tiny 2-layer jet correction (thermal-wind inspired) ---
    # Real jets strengthen where meridional temperature gradients are strong (mid-lats).
    # We approximate an upper-level westerly anomaly from |dT/dy| and mix a fraction down.
    # Use the same metric as the pressure-gradient step above (meters per latitude row).
    dT_dy = np.gradient(T, axis=0) / dy
    jet_window = np.exp(-((abs_deg[:, None] - 45.0) / 18.0) ** 2).astype(np.float32)
    thermal_wind_coeff = 2.0e6  # tuned: |dT/dy|~1e-5 K/m -> ~20 m/s aloft
    u_aloft = thermal_wind_coeff * jet_window * np.abs(dT_dy)
    surface_mix = 0.20
    uc = uc + surface_mix * u_aloft
    lat_amp = 0.06 + 0.60 * (lat_factor ** 1.6)
    global_amp = 0.30
    uc = uc * lat_amp[:, None] * global_amp
    vc = vc * lat_amp[:, None] * global_amp
    
    # Convert to vorticity and solve via streamfunction to ensure divergence-free
    dvc_dx = _ddx_periodic(vc)
    duc_dy = -np.gradient(uc, axis=0)
    omega = dvc_dx - duc_dy
    
    # Solve for streamfunction
    psi = _streamfunction_from_vorticity(omega)
    u_stream = -np.gradient(psi, axis=0) * (Hc / (np.pi))
    v_stream = -_ddx_periodic(psi) * (Wc / (2.0 * np.pi))
    
    # Blend: mostly from pressure gradients, streamfunction ensures consistency
    uc = 0.75 * uc + 0.25 * u_stream
    vc = 0.75 * vc + 0.25 * v_stream
    
    # Apply terrain effects: blocking, channeling, deflection
    if terrain_influence > 0:
        elev_norm = np.clip(elev_c, 0.0, 1.0)
        tf = float(np.clip(terrain_flow_amp, 0.0, 1.0))
        
        # Mountain blocking
        block_factor = np.clip(1.0 - tf * terrain_influence * 0.5 * elev_norm, 0.6, 1.0)
        
        # Terrain channeling (flow follows valleys) - reduced from 0.25 to 0.15
        slope_mag = np.hypot(gx, gy)
        channel_factor = 1.0 + tf * terrain_influence * 0.08 * slope_mag
        
        # Deflection around obstacles
        deflect_u = -tf * terrain_influence * 0.12 * gx * slope_mag
        deflect_v = -tf * terrain_influence * 0.12 * gy * slope_mag
        
        uc = (uc * block_factor + deflect_u) * channel_factor
        vc = (vc * block_factor + deflect_v) * channel_factor
        
        # Land-sea friction contrast (keep subtle; strong contrast creates stationary speed blobs)
        friction_factor = np.where(sea_mask, 1.03, 0.97)
        uc *= friction_factor
        vc *= friction_factor
    
    # Final smoothing
    uc = uc + 0.10 * _laplacian(uc)
    vc = vc + 0.10 * _laplacian(vc)
    
    uc = np.clip(uc, -u_limit[:, None], u_limit[:, None]).astype(np.float32)
    vc = np.clip(vc, -v_limit[:, None], v_limit[:, None]).astype(np.float32)
    
    # Debug logging for wind diagnostics
    if debug_log:
        from terrain import LOG
        wind_mag = np.sqrt(uc*uc + vc*vc)
        LOG.info(f"[Wind Debug Day {day_of_year}]")
        LOG.info(f"  Pressure: min={float(np.min(pressure)):.1f}, mean={float(np.mean(pressure)):.1f}, max={float(np.max(pressure)):.1f} hPa")
        LOG.info(f"  Pressure gradients: dp_dx mean={float(np.mean(np.abs(dp_dx))):.4f}, dp_dy mean={float(np.mean(np.abs(dp_dy))):.4f}")
        LOG.info(f"  f_coriolis: min={float(np.min(np.abs(f_coriolis))):.2e}, max={float(np.max(np.abs(f_coriolis))):.2e}")
        LOG.info(f"  u_final: min={float(np.min(uc)):.1f}, mean={float(np.mean(uc)):.1f}, max={float(np.max(uc)):.1f} m/s")
        LOG.info(f"  v_final: min={float(np.min(vc)):.1f}, mean={float(np.mean(vc)):.1f}, max={float(np.max(vc)):.1f} m/s")
        LOG.info(f"  Wind magnitude: min={float(np.min(wind_mag)):.1f}, mean={float(np.mean(wind_mag)):.1f}, max={float(np.max(wind_mag)):.1f} m/s")
        LOG.info(f"  Wind percentiles: p10={float(np.percentile(wind_mag, 10)):.1f}, p50={float(np.percentile(wind_mag, 50)):.1f}, p90={float(np.percentile(wind_mag, 90)):.1f} m/s")
        
        # Latitude band breakdown
        eq_band = np.abs(lat) < np.deg2rad(10)
        trop_band = (np.abs(lat) >= np.deg2rad(10)) & (np.abs(lat) < np.deg2rad(30))
        mid_band = (np.abs(lat) >= np.deg2rad(30)) & (np.abs(lat) < np.deg2rad(60))
        
        LOG.info(f"  By latitude - Equatorial (0-10°): mean={float(np.mean(wind_mag[eq_band[:, None].repeat(Wc, 1)])):.1f} m/s")
        LOG.info(f"  By latitude - Tropical (10-30°): mean={float(np.mean(wind_mag[trop_band[:, None].repeat(Wc, 1)])):.1f} m/s")
        LOG.info(f"  By latitude - Mid-lat (30-60°): mean={float(np.mean(wind_mag[mid_band[:, None].repeat(Wc, 1)])):.1f} m/s")
        
        # Clipping statistics
        u_clipped = np.sum((uc == 30.0) | (uc == -30.0))
        v_clipped = np.sum((vc == 20.0) | (vc == -20.0))
        total_cells = uc.size
        LOG.info(f"  Clipping: u_clipped={u_clipped}/{total_cells} ({100.0*u_clipped/total_cells:.1f}%), v_clipped={v_clipped}/{total_cells} ({100.0*v_clipped/total_cells:.1f}%)")

    up = _upsample_bilinear if str(upsample).lower() == "bilinear" else _upsample_repeat
    return up(uc, H, W, block_size), up(vc, H, W, block_size)


def _cos_window(x_deg: np.ndarray, a: float, b: float) -> np.ndarray:
    # Smooth 0→1→0 window over [a,b] degrees using raised cosine; outside=0.
    x = np.asarray(x_deg, dtype=np.float32)
    y = np.zeros_like(x)
    m = (x >= a) & (x <= b)
    t = (x[m] - a) / max(b - a, 1e-6)
    y[m] = 0.5 * (1.0 - np.cos(np.pi * t))
    return y


def render_wind_arrows(height: int, width: int, u: np.ndarray, v: np.ndarray, *, step: int | None = None, target_arrows: int | None = 250, scale: float = 0.8) -> np.ndarray:
    """Rasterize sparse white arrows onto black background; returns (H,W,3) float.

    - `step` controls arrow density; `scale` scales arrow length relative to step.
    - Arrows point in wind direction; length varies slightly with wind speed for readability.
    """
    H = int(height); W = int(width)
    img = np.zeros((H, W, 3), dtype=np.float32)
    mag = np.sqrt(u * u + v * v) + 1e-9
    umax = np.percentile(mag, 95.0) + 1e-6
    if step is None:
        # Choose step so ~target_arrows triangles are drawn on equirectangular grid
        n_target = max(50, int(target_arrows or 250))
        step_f = np.sqrt((H * W) / float(n_target))
        sx = sy = max(6, int(step_f))
    else:
        sx = sy = max(4, int(step))
    step_len = float(min(sx, sy))
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for y in range(sy // 2, H, sy):
        for x in range(sx // 2, W, sx):
            uu = float(u[y, x]); vv = float(v[y, x])
            m = np.sqrt(uu * uu + vv * vv)
            if m < 1e-3:
                continue
            # Direction vector (normalized). Note: screen y grows downward, so flip v.
            dx = uu / m
            dy = -vv / m  # screen y grows down

            # Mostly-constant arrow size (matches "quiver"-like look); small speed modulation.
            t = min(1.0, m / umax)
            arrow_len = scale * step_len * (0.42 + 0.22 * t)
            arrow_len = max(3.0, min(arrow_len, step_len * 0.85))

            # Shaft from tail -> just before head.
            head_len = max(2.0, arrow_len * 0.45)
            shaft_len = max(1.0, arrow_len - head_len)

            tail_x = x - dx * (shaft_len * 0.55)
            tail_y = y - dy * (shaft_len * 0.55)
            head_base_x = x + dx * (shaft_len * 0.45)
            head_base_y = y + dy * (shaft_len * 0.45)
            tip_x = head_base_x + dx * head_len
            tip_y = head_base_y + dy * head_len

            _draw_line(img, tail_x, tail_y, head_base_x, head_base_y, white)

            # Arrow head: narrow filled triangle.
            perp_x = -dy
            perp_y = dx
            head_w = max(1.0, head_len * 0.55)
            p1_x = head_base_x + perp_x * head_w
            p1_y = head_base_y + perp_y * head_w
            p2_x = head_base_x - perp_x * head_w
            p2_y = head_base_y - perp_y * head_w
            _draw_triangle(img, tip_x, tip_y, p1_x, p1_y, p2_x, p2_y, white)
    return img


def wind_speed_to_rgb(
    speed: np.ndarray,
    *,
    vmax: float | None = None,
    gamma: float = 0.75,
) -> np.ndarray:
    """Map wind speed (m/s) -> RGB float image (H,W,3).

    If `vmax` is None, choose a robust per-frame scale from the data
    (99.5th percentile). This avoids fixed ceilings and lets the colormap
    adapt to whatever range the simulation produces.
    """
    s = speed.astype(np.float32)
    if vmax is None:
        # Robust scale: ignore rare extremes so the map doesn't saturate.
        vm = float(np.nanpercentile(s, 99.5))
    else:
        vm = float(vmax)
    vm = max(1e-6, vm)
    t = np.clip(s / vm, 0.0, 1.0)
    t = t ** float(gamma)

    # Slow → fast: blue → green → yellow → red.
    cstops = np.array(
        [
            [0.06, 0.25, 0.92],  # blue (slow)
            [0.10, 0.78, 0.55],  # green
            [0.95, 0.88, 0.20],  # yellow
            [0.92, 0.18, 0.10],  # red (fast)
        ],
        dtype=np.float32,
    )
    bp = np.array([0.0, 0.50, 0.80, 1.0], dtype=np.float32)
    i = np.clip(np.searchsorted(bp, t, side="right") - 1, 0, len(bp) - 2)
    c0 = cstops[i]
    c1 = cstops[i + 1]
    tt = (t - bp[i]) / (bp[i + 1] - bp[i] + 1e-6)
    return c0 + (c1 - c0) * tt[..., None]


def _draw_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, col: np.ndarray) -> None:
    H, W, _ = img.shape
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for i in range(n):
        t = i / max(n - 1, 1)
        xi = int(round(x0 + (x1 - x0) * t))
        yi = int(round(y0 + (y1 - y0) * t))
        if 0 <= xi < W and 0 <= yi < H:
            img[yi, xi, :] = np.maximum(img[yi, xi, :], col)


def _draw_triangle(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, x2: float, y2: float, col: np.ndarray) -> None:
    """Draw filled triangle using scanline algorithm."""
    H, W, _ = img.shape
    # Sort vertices by y
    pts = [(x0, y0), (x1, y1), (x2, y2)]
    pts.sort(key=lambda p: p[1])
    x0, y0 = pts[0]; x1, y1 = pts[1]; x2, y2 = pts[2]
    
    # Bounding box
    min_x = int(max(0, min(x0, x1, x2)))
    max_x = int(min(W - 1, max(x0, x1, x2)))
    min_y = int(max(0, min(y0, y1, y2)))
    max_y = int(min(H - 1, max(y0, y1, y2)))
    
    if min_y >= max_y or min_x >= max_x:
        return
    
    # Scanline fill
    for y in range(min_y, max_y + 1):
        # Find intersections with triangle edges
        intersections = []
        # Edge 0-1
        if y0 != y1:
            t = (y - y0) / (y1 - y0)
            if 0 <= t <= 1:
                intersections.append(x0 + t * (x1 - x0))
        # Edge 0-2
        if y0 != y2:
            t = (y - y0) / (y2 - y0)
            if 0 <= t <= 1:
                intersections.append(x0 + t * (x2 - x0))
        # Edge 1-2
        if y1 != y2:
            t = (y - y1) / (y2 - y1)
            if 0 <= t <= 1:
                intersections.append(x1 + t * (x2 - x1))
        
        if len(intersections) >= 2:
            x_start = int(round(min(intersections)))
            x_end = int(round(max(intersections)))
            for x in range(max(min_x, x_start), min(max_x + 1, x_end + 1)):
                if 0 <= x < W and 0 <= y < H:
                    img[y, x, :] = np.maximum(img[y, x, :], col)


def _speed_color(t: float) -> np.ndarray:
    # Blue→Red gradient by speed fraction t∈[0,1]
    t = float(np.clip(t, 0.0, 1.0))
    c0 = np.array([0.05, 0.40, 1.00], dtype=np.float32)
    c1 = np.array([1.00, 0.20, 0.05], dtype=np.float32)
    return (1.0 - t) * c0 + t * c1


def generate_precipitation(
    height: int,
    width: int,
    elevation: np.ndarray,
    *,
    temperature: np.ndarray | None = None,
    wind_u: np.ndarray | None = None,
    wind_v: np.ndarray | None = None,
    humidity: np.ndarray | None = None,
    soil_moisture: np.ndarray | None = None,
    day_of_year: int = 80,
    dt_days: float = 1.0,
    evap_coeff: float = 1.0,
    uplift_coeff: float = 1.0,
    rain_efficiency: float = 0.7,
    target_mean_mm_day: float = 2.7,
    max_precip_mm_day: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precip_mm_day, humidity, soil_moisture).

    The model keeps a prognostic surface humidity field and a simple soil-moisture
    bucket while blending three precipitation triggers: moisture convergence,
    orographic lift, and convective instability. Everything runs at the native
    grid resolution so it can operate in both snapshot and time-stepping modes.
    """

    H = int(height)
    W = int(width)
    elev = elevation.astype(np.float32)

    land_mask, sea_mask = _derive_land_sea_masks(elev)
    land_f = land_mask.astype(np.float32)
    sea_f = sea_mask.astype(np.float32)

    if temperature is None:
        lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
        T_lat = temperature_kelvin_for_lat(lat, day_of_year=day_of_year)
        temperature = np.repeat(T_lat[:, None], W, axis=1).astype(np.float32)
    else:
        temperature = temperature.astype(np.float32)

    u: np.ndarray
    v: np.ndarray
    if wind_u is None or wind_v is None:
        u, v = generate_wind_field(
            H,
            W,
            day_of_year=day_of_year,
            block_size=1,
            elevation=elev,
            debug_log=False,
        )
    else:
        u = wind_u.astype(np.float32)
        v = wind_v.astype(np.float32)

    wind_speed = np.sqrt(u * u + v * v) + 1e-6
    temp_norm = np.clip((temperature - 255.0) / 45.0, 0.0, 1.0)

    Tc = np.clip(temperature - 273.15, -60.0, 60.0)
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    qsat = np.clip(0.622 * es / 1013.25, 0.0, 0.035).astype(np.float32)

    if humidity is None:
        base_q = np.where(sea_mask, 0.013, 0.009).astype(np.float32)
    else:
        base_q = humidity.astype(np.float32)

    if soil_moisture is None:
        soil = np.where(land_mask, 0.55, 0.0).astype(np.float32)
    else:
        soil = soil_moisture.astype(np.float32)

    dt = max(float(dt_days), 1.0)

    # Evaporation and evapotranspiration sources
    wind_norm = np.clip(wind_speed / 15.0, 0.0, 1.5)
    ocean_evap = evap_coeff * sea_f * (0.45 + 0.55 * wind_norm) * np.clip(qsat - base_q, 0.0, None)
    land_evap = (
        evap_coeff
        * land_f
        * (0.20 + 0.65 * temp_norm)
        * (0.35 + 0.65 * soil)
        * np.clip(qsat - base_q, 0.0, None)
    )
    sources = (ocean_evap + land_evap) * dt
    q = np.clip(base_q + sources, 0.0, qsat)

    # Moisture advection/diffusion (semi-Lagrangian-ish)
    u_scale = np.clip(np.abs(u) / 20.0, 0.0, 1.0) * 0.4
    v_scale = np.clip(np.abs(v) / 12.0, 0.0, 1.0) * 0.4
    for _ in range(3):
        q = _advect_scalar(q, u, v, u_scale, v_scale)
        q = q + 0.12 * _laplacian(q)
        q = np.clip(q, 0.0, qsat)

    # Moisture-flux convergence driver
    flux_x = q * u
    flux_y = q * v
    conv = np.clip(-(_ddx_periodic(flux_x) + np.gradient(flux_y, axis=0)), 0.0, None)
    conv = conv / (np.mean(conv) + 1e-6)
    conv = np.clip(conv + 0.15 * _laplacian(conv), 0.0, 3.0)

    # Large-scale ascent proxy from wind convergence
    div = _ddx_periodic(u) + np.gradient(v, axis=0)
    ascent = np.clip(-div, 0.0, None)
    ascent = ascent / (np.mean(ascent) + 1e-6)
    ascent = np.clip(ascent + 0.15 * _laplacian(ascent), 0.0, 3.0)

    # Orographic uplift signal
    gx = _ddx_periodic(elev)
    gy = np.gradient(elev, axis=0)
    slope = np.hypot(gx, gy)
    orog = np.clip(gx * u + gy * v, 0.0, None) + 0.25 * slope
    orog = land_f * orog
    orog = orog / (np.percentile(orog, 90.0) + 1e-6)
    orog = np.clip(orog + 0.15 * _laplacian(orog), 0.0, 2.0)

    # Convective instability proxy
    rh = q / (qsat + 1e-6)
    convective = np.clip((temp_norm - 0.4) * rh, 0.0, None)
    convective = np.clip(convective + 0.1 * conv, 0.0, 2.0)

    # Blend drivers into precipitation potential
    precip_potential = uplift_coeff * (
        0.40 * rh +
        0.20 * conv +
        0.20 * orog +
        0.15 * convective +
        0.15 * ascent
    )
    for _ in range(3):
        precip_potential = np.clip(precip_potential + 0.18 * _laplacian(precip_potential), 0.0, 3.0)

    # Convert potential to precipitation (mm/day) with moisture conservation
    remove_frac = np.clip(rain_efficiency * precip_potential * dt, 0.0, 1.0)
    dq = np.clip(remove_frac * q, 0.0, q)
    column_mm_per_q = 2000.0  # ~20 mm PW for q=0.01
    P = dq * (column_mm_per_q / dt)
    if target_mean_mm_day > 0.0:
        mean_p = float(np.mean(P))
        scale = float(np.clip(target_mean_mm_day / (mean_p + 1e-6), 0.2, 3.0))
        dq = np.clip(dq * scale, 0.0, q)
        P = dq * (column_mm_per_q / dt)
    if max_precip_mm_day > 0.0:
        cap = np.minimum(1.0, max_precip_mm_day / (P + 1e-9))
        dq = dq * cap
        P = P * cap

    # Update humidity and soil moisture reservoirs
    humidity_next = np.clip(q - dq, 0.0, qsat)

    soil += (P * land_f) * 0.0006 - (land_evap * dt) * 0.4
    soil = np.where(land_mask, np.clip(soil, 0.05, 1.0), 0.0)

    return (
        P.astype(np.float32),
        humidity_next.astype(np.float32),
        soil.astype(np.float32),
    )
