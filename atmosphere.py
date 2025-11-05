"""Simple 3-cell-per-hemisphere wind model for equirectangular maps.

Hadley, Ferrel, and Polar cells are approximated by prescribing zonal (u)
and meridional (v) surface winds by latitude, with Coriolis turning that
creates easterlies and westerlies in the appropriate bands.

This is intentionally lightweight for interactive use; it returns a dense
vector field and a pre-rendered arrow RGB overlay for display.
"""

from __future__ import annotations

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
    
    # Create coordinate arrays for interpolation
    y_coarse = np.arange(Hc, dtype=np.float32)
    x_coarse = np.arange(Wc, dtype=np.float32)
    y_fine = np.linspace(0, Hc - 1, H, dtype=np.float32)
    x_fine = np.linspace(0, Wc - 1, W, dtype=np.float32)
    
    # Create meshgrid for fine coordinates
    Y_fine, X_fine = np.meshgrid(y_fine, x_fine, indexing='ij')
    
    # Find integer indices and fractional parts for bilinear interpolation
    y_idx = np.floor(Y_fine).astype(np.int32)
    x_idx = np.floor(X_fine).astype(np.int32)
    y_frac = Y_fine - y_idx
    x_frac = X_fine - x_idx
    
    # Clamp indices to valid range
    y_idx = np.clip(y_idx, 0, Hc - 1)
    x_idx = np.clip(x_idx, 0, Wc - 1)
    y_idx_next = np.clip(y_idx + 1, 0, Hc - 1)
    x_idx_next = np.clip(x_idx + 1, 0, Wc - 1)
    
    # Bilinear interpolation: interpolate in x first, then y
    # Top edge
    f_top_left = field[y_idx, x_idx]
    f_top_right = field[y_idx, x_idx_next]
    f_top = f_top_left * (1.0 - x_frac) + f_top_right * x_frac
    
    # Bottom edge
    f_bot_left = field[y_idx_next, x_idx]
    f_bot_right = field[y_idx_next, x_idx_next]
    f_bot = f_bot_left * (1.0 - x_frac) + f_bot_right * x_frac
    
    # Interpolate in y
    result = (f_top * (1.0 - y_frac) + f_bot * y_frac).astype(np.float32)
    return result


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
    thresh = float(np.median(elev))
    land = elev > thresh
    land = _majority_filter(land, iterations=2)
    sea = ~land
    return land, sea


def _laplacian(field: np.ndarray) -> np.ndarray:
    pad = np.pad(field, 1, mode="edge")
    return (
        pad[0:-2, 1:-1]
        + pad[2:, 1:-1]
        + pad[1:-1, 0:-2]
        + pad[1:-1, 2:]
        - 4.0 * field
    )


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


def generate_wind_field(
    height: int,
    width: int,
    *,
    day_of_year: int = 80,
    block_size: int = 3,
    elevation: np.ndarray | None = None,
    terrain_influence: float = 1.0,
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

    # Build 2D temperature field with land-sea contrast and longitudinal variation
    T_lat = temperature_kelvin_for_lat(lat, day_of_year=day_of_year).astype(np.float32)
    T = np.repeat(T_lat[:, None], Wc, axis=1).astype(np.float32)
    
    # Land-sea temperature contrast (land warmer in summer, cooler in winter)
    land_f = land_mask.astype(np.float32)
    sea_f = sea_mask.astype(np.float32)
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
    T = T + 0.15 * _laplacian(T)
    T = np.clip(T, 200.0, 320.0)
    
    # Convert temperature to surface pressure (ideal gas law approximation)
    # Warmer = lower pressure (thermal low), colder = higher pressure (thermal high)
    T_ref = 273.15
    p_thermal = 1013.25 * (T_ref / T) ** 3.5  # hPa, exaggerated for stronger gradients
    
    # Add terrain pressure anomalies (mountains create blocking highs)
    p_terrain = 25.0 * terrain_influence * np.clip(elev_c, 0.0, 1.0)
    
    # Inject weather system perturbations (quasi-stationary cyclones/anticyclones)
    rng = np.random.default_rng(int(day_of_year) + 9001)
    n_systems = 8
    for _ in range(n_systems):
        sys_lon = rng.uniform(-np.pi, np.pi)
        sys_lat = rng.uniform(-0.6 * np.pi, 0.6 * np.pi)
        sys_strength = rng.uniform(-18.0, 18.0)
        sys_scale = rng.uniform(0.15, 0.35)
        
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
    f_min = 1e-5
    mask_pos = f_coriolis >= 0
    f_coriolis = np.where(mask_pos, np.maximum(f_coriolis, f_min), np.minimum(f_coriolis, -f_min))
    
    dp_dy = np.gradient(pressure, axis=0) / (111320.0 / Hc)  # Pa/m (approximate)
    dp_dx = np.gradient(pressure, axis=1) / (111320.0 * np.cos(lat[:, None]) / Wc)  # Pa/m
    
    # Geostrophic wind (m/s)
    rho = 1.2  # kg/m³ air density
    u_geo = -(1.0 / (rho * f_coriolis[:, None])) * dp_dy
    v_geo = (1.0 / (rho * f_coriolis[:, None])) * dp_dx
    
    # Scale to realistic speeds (geostrophic approximation overestimates near surface)
    u_geo = np.clip(u_geo * 0.6, -25.0, 25.0)
    v_geo = np.clip(v_geo * 0.6, -15.0, 15.0)
    
    # Add ageostrophic component (cross-isobar flow toward low pressure)
    ageo_frac = 0.15
    u_ageo = -ageo_frac * dp_dx / (np.abs(dp_dx) + 1e-3) * np.abs(u_geo) * 0.3
    v_ageo = -ageo_frac * dp_dy / (np.abs(dp_dy) + 1e-3) * np.abs(v_geo) * 0.3
    
    uc = u_geo + u_ageo
    vc = v_geo + v_ageo
    
    # Convert to vorticity and solve via streamfunction to ensure divergence-free
    dvc_dx = np.gradient(vc, axis=1)
    duc_dy = np.gradient(uc, axis=0)
    omega = dvc_dx - duc_dy
    
    # Solve for streamfunction
    psi = _streamfunction_from_vorticity(omega)
    u_stream = np.gradient(psi, axis=0) * (Hc / (np.pi))
    v_stream = -np.gradient(psi, axis=1) * (Wc / (2.0 * np.pi))
    
    # Blend: mostly from pressure gradients, streamfunction ensures consistency
    uc = 0.75 * uc + 0.25 * u_stream
    vc = 0.75 * vc + 0.25 * v_stream
    
    # Apply terrain effects: blocking, channeling, deflection
    if terrain_influence > 0:
        elev_norm = np.clip(elev_c, 0.0, 1.0)
        
        # Mountain blocking
        block_factor = np.clip(1.0 - terrain_influence * 0.5 * elev_norm, 0.4, 1.0)
        
        # Terrain channeling (flow follows valleys)
        slope_mag = np.hypot(gx, gy)
        channel_factor = 1.0 + terrain_influence * 0.25 * slope_mag
        
        # Deflection around obstacles
        deflect_u = -terrain_influence * 0.3 * gx * slope_mag
        deflect_v = -terrain_influence * 0.3 * gy * slope_mag
        
        uc = (uc * block_factor + deflect_u) * channel_factor
        vc = (vc * block_factor + deflect_v) * channel_factor
        
        # Land-sea friction contrast
        friction_factor = np.where(sea_mask, 1.15, 0.85)
        uc *= friction_factor
        vc *= friction_factor
    
    # Final smoothing
    uc = uc + 0.10 * _laplacian(uc)
    vc = vc + 0.10 * _laplacian(vc)
    
    uc = np.clip(uc, -30.0, 30.0).astype(np.float32)
    vc = np.clip(vc, -20.0, 20.0).astype(np.float32)

    return _upsample_repeat(uc, H, W, block_size), _upsample_repeat(vc, H, W, block_size)


def _cos_window(x_deg: np.ndarray, a: float, b: float) -> np.ndarray:
    # Smooth 0→1→0 window over [a,b] degrees using raised cosine; outside=0.
    x = np.asarray(x_deg, dtype=np.float32)
    y = np.zeros_like(x)
    m = (x >= a) & (x <= b)
    t = (x[m] - a) / max(b - a, 1e-6)
    y[m] = 0.5 * (1.0 - np.cos(np.pi * t))
    return y


def render_wind_arrows(height: int, width: int, u: np.ndarray, v: np.ndarray, *, step: int | None = None, target_arrows: int | None = 250, scale: float = 0.8) -> np.ndarray:
    """Rasterize sparse white triangles onto black background; returns (H,W,3) float.

    - `step` controls triangle density; `scale` scales triangle size relative to step.
    - Triangles point in wind direction, size scales with wind speed.
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
            # Triangle size scales with wind speed
            size = scale * (m / umax) * step_len * 0.5
            size = max(2.0, min(size, step_len * 0.8))
            # Direction vector (normalized)
            dx = uu / m
            dy = -vv / m  # screen y grows down
            # Triangle points: tip in wind direction, base perpendicular
            tip_x = x + dx * size * 0.6
            tip_y = y + dy * size * 0.6
            base_x = x - dx * size * 0.4
            base_y = y - dy * size * 0.4
            # Perpendicular direction for base
            perp_x = -dy
            perp_y = dx
            base_w = size * 0.4
            p1_x = base_x + perp_x * base_w
            p1_y = base_y + perp_y * base_w
            p2_x = base_x - perp_x * base_w
            p2_y = base_y - perp_y * base_w
            _draw_triangle(img, tip_x, tip_y, p1_x, p1_y, p2_x, p2_y, white)
    return img


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
    conv = np.clip(-(np.gradient(flux_x, axis=1) + np.gradient(flux_y, axis=0)), 0.0, None)
    conv = conv / (np.mean(conv) + 1e-6)
    conv = np.clip(conv + 0.15 * _laplacian(conv), 0.0, 3.0)

    # Orographic uplift signal
    gx, gy = np.gradient(elev)
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
        0.45 * rh +
        0.25 * conv +
        0.20 * orog +
        0.20 * convective
    )
    for _ in range(3):
        precip_potential = np.clip(precip_potential + 0.18 * _laplacian(precip_potential), 0.0, 3.0)

    # Convert potential to precipitation (mm/day)
    precip_rate = rain_efficiency * precip_potential * q
    P = np.clip(480.0 * precip_rate, 0.0, None)

    # Update humidity and soil moisture reservoirs
    rain_sink = np.clip(P * 0.00004, 0.0, q)
    humidity_next = np.clip(q - rain_sink, 0.0, qsat)

    soil += (P * land_f) * 0.0006 - (land_evap * dt) * 0.4
    soil = np.where(land_mask, np.clip(soil, 0.05, 1.0), 0.0)

    return (
        P.astype(np.float32),
        humidity_next.astype(np.float32),
        soil.astype(np.float32),
    )


