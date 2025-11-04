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


def generate_wind_field(height: int, width: int, *, day_of_year: int = 80, block_size: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) winds at surface on equirectangular grid (H,W).

    u: eastward (m/s), v: northward (m/s). Positive u is to the right in map.

    Model:
    - Define three latitude bands per hemisphere: Polar (60–90), Ferrel (30–60), Hadley (0–30).
    - Base zonal speeds (Earth-like, near-surface):
        Polar easterlies ~ 6 m/s, Mid-lat westerlies ~ 12 m/s, Trade winds ~ 8 m/s.
    - Smooth with cos windows at band edges. Meridional component small return flow.
    - Reverse sign across equator for symmetry.
    """
    H = int(height); W = int(width)
    Hc, Wc = _coarse_shape(H, W, block_size)
    lat = _latitudes_h(Hc)
    abs_deg = np.rad2deg(np.abs(lat))

    # Meridional temperature gradient magnitude as simple driver (Hadley strength)
    T = temperature_kelvin_for_lat(lat, day_of_year=day_of_year).astype(np.float32)
    dtheta = np.pi / float(H)
    dT_dy = np.gradient(T, dtheta)  # K per radian northward
    grad_norm = np.abs(dT_dy)
    gmax = float(np.max(grad_norm) + 1e-6)
    strength = 0.6 + 0.8 * (grad_norm / gmax)  # 0.6..1.4 multiplier

    # Base zonal profiles (per-latitude), sign by hemisphere via coriolis turning
    # Trades: easterlies 0–30 → u < 0 in both hemispheres
    w_trade = 8.0 * _cos_window(abs_deg, 0.0, 30.0)
    u_trade = -np.sign(lat) * 0.0  # sign handled below with easterly convention
    u_trade = -w_trade  # easterlies → negative u

    # Westerlies: 30–60 → u > 0
    w_west = 12.0 * _cos_window(abs_deg, 30.0, 60.0)
    u_west = +w_west

    # Polar easterlies: 60–90 → u < 0
    w_polar = 6.0 * _cos_window(abs_deg, 60.0, 90.0)
    u_polar = -w_polar

    # Total zonal; taper to zero at equator to avoid discontinuity
    u_lat = (u_trade + u_west + u_polar) * strength
    u_lat *= (1.0 - 0.2 * np.cos(np.clip(abs_deg, 0.0, 5.0) / 5.0 * np.pi))  # tiny equator reduction

    # Meridional return flows (small). Hadley near-surface toward equator (v<0 in N, v>0 in S)
    v_hadley = -2.0 * np.sign(lat) * _cos_window(abs_deg, 5.0, 30.0)
    # Ferrel indirect cell: surface poleward (v>0 in N, v<0 in S) around 30–60
    v_ferrel = +1.0 * np.sign(lat) * _cos_window(abs_deg, 30.0, 60.0)
    # Polar cell: surface toward equator (v<0 in N, v>0 in S) 60–85
    v_polar = -0.5 * np.sign(lat) * _cos_window(abs_deg, 60.0, 85.0)
    v_lat = (v_hadley + v_ferrel + v_polar) * (0.8 + 0.4 * (grad_norm / gmax))

    # Broadcast to full field (assume longitude-uniform mean winds)
    uc = np.repeat(u_lat[:, None], Wc, axis=1).astype(np.float32)
    vc = np.repeat(v_lat[:, None], Wc, axis=1).astype(np.float32)
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
    """Rasterize sparse white arrows onto black background; returns (H,W,3) float.

    - `step` controls arrow density; `scale` scales vector length relative to step.
    - Uses simple Bresenham-like interpolation for the arrow shaft; small triangle head.
    """
    H = int(height); W = int(width)
    img = np.zeros((H, W, 3), dtype=np.float32)
    mag = np.sqrt(u * u + v * v) + 1e-9
    umax = np.percentile(mag, 95.0) + 1e-6
    if step is None:
        # Choose step so ~target_arrows arrows are drawn on equirectangular grid
        n_target = max(50, int(target_arrows or 250))
        step_f = np.sqrt((H * W) / float(n_target))
        sx = sy = max(6, int(step_f))
    else:
        sx = sy = max(4, int(step))
    step_len = float(min(sx, sy))
    for y in range(sy // 2, H, sy):
        for x in range(sx // 2, W, sx):
            uu = float(u[y, x]); vv = float(v[y, x])
            m = np.sqrt(uu * uu + vv * vv)
            if m < 1e-3:
                continue
            L = scale * (m / umax) * 0.9 * step_len
            dx = uu / m * L
            dy = -vv / m * L  # screen y grows down
            x0, y0 = x - dx * 0.5, y - dy * 0.5
            x1, y1 = x + dx * 0.5, y + dy * 0.5
            col = _speed_color(m / umax)
            _draw_line(img, x0, y0, x1, y1, col)
            _draw_head(img, x1, y1, dx, dy, col)
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


def _draw_head(img: np.ndarray, x1: float, y1: float, dx: float, dy: float, col: np.ndarray) -> None:
    H, W, _ = img.shape
    nx, ny = -dy, dx
    nx, ny = nx / (np.hypot(nx, ny) + 1e-9), ny / (np.hypot(nx, ny) + 1e-9)
    back_x = x1 - 0.25 * dx
    back_y = y1 - 0.25 * dy
    p0 = (int(round(x1)), int(round(y1)))
    p1 = (int(round(back_x + 0.12 * dx + 0.18 * nx * np.hypot(dx, dy))), int(round(back_y + 0.12 * dy + 0.18 * ny * np.hypot(dx, dy))))
    p2 = (int(round(back_x + 0.12 * dx - 0.18 * nx * np.hypot(dx, dy))), int(round(back_y + 0.12 * dy - 0.18 * ny * np.hypot(dx, dy))))
    for (xa, ya), (xb, yb) in [(p0, p1), (p0, p2), (p1, p2)]:
        _draw_line(img, xa, ya, xb, yb, col)


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
    day_of_year: int = 80,
    evap_coeff: float = 1.0,
    uplift_coeff: float = 1.0,
    rain_efficiency: float = 0.7,
    iterations: int = 48,
    wind_ref: float = 12.0,
    seed: int | None = None,
    block_size: int = 2,
    evap_land_coeff: float = 0.2,
    recycle_coeff: float = 0.2,
    conv_eff: float = 0.4,
    conv_rh0: float = 0.8,
    kdiff: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (precip_mm_day, humidity) using simple moisture budget.

    - Sea mask from median elevation.
    - q_sat from temperature; evaporation over ocean; condensation with convergence
      and orographic uplift; advection via simple upwind relaxation against winds.
    """
    H = int(height); W = int(width)
    bs = max(1, int(block_size))
    Hc, Wc = _coarse_shape(H, W, bs)
    # Downsample elevation by block mean
    elev_pad = np.pad(elevation.astype(np.float32), ((0, Hc*bs - H), (0, Wc*bs - W)), mode="edge")
    elev_c = elev_pad.reshape(Hc, bs, Wc, bs).mean(axis=(1, 3))
    sea_c = elev_c <= float(np.median(elev_c))
    land_c = ~sea_c
    lat_c = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    T_c = np.repeat(temperature_kelvin_for_lat(lat_c, day_of_year=day_of_year)[:, None], Wc, axis=1).astype(np.float32)
    Tc = np.clip(T_c - 273.15, -100.0, 100.0)  # Clip to prevent overflow in exp
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))  # hPa
    qsat = np.clip(0.622 * es / 1013.25, 0.0, 0.03).astype(np.float32)
    u, v = generate_wind_field(Hc, Wc, day_of_year=day_of_year, block_size=1)
    mag = np.sqrt(u * u + v * v) + 1e-6
    q = (sea_c.astype(np.float32) * 0.5 * qsat).astype(np.float32)
    P = np.zeros((Hc, Wc), dtype=np.float32)
    dt = 1.0 / max(1, int(iterations))
    ufac = np.clip(mag / max(wind_ref, 1e-3), 0.0, 1.0)
    gx, gy = np.gradient(elev_c)
    # Latitude weighting for deep convection (ITCZ-like)
    lat_w = (np.cos(lat_c) ** 2)[:, None].astype(np.float32)

    def _lap(a: np.ndarray) -> np.ndarray:
        ap = np.pad(a, 1, mode="edge")
        return ap[0:-2,1:-1] + ap[2:,1:-1] + ap[1:-1,0:-2] + ap[1:-1,2:] - 4.0 * a

    for _ in range(int(iterations)):
        # Ocean + small land evap
        mask = sea_c.astype(np.float32) + float(evap_land_coeff) * land_c.astype(np.float32)
        E = evap_coeff * mask * np.clip(qsat - q, 0.0, None) * (0.6 + 0.4 * ufac)
        # Moisture-flux convergence (adds longitudinal variability)
        mx = q * u; my = q * v
        dmxdx = np.gradient(mx, axis=1); dmydy = np.gradient(my, axis=0)
        conv = np.clip(-(dmxdx + dmydy), 0.0, None)
        conv /= (np.percentile(conv, 95.0) + 1e-6)
        orog = np.clip(gx * u + gy * v, 0.0, None)
        orog /= (np.percentile(orog, 95.0) + 1e-6)
        widx = uplift_coeff * (conv + 0.5 * orog)
        widx = np.clip(widx, 0.0, 2.0)
        # Large-scale + convective rain
        r_ls = rain_efficiency * widx * q
        rh = q / (qsat + 1e-6)
        r_cv = conv_eff * lat_w * np.clip(rh - conv_rh0, 0.0, None) * q
        r = r_ls + r_cv
        r = np.minimum(r, q / dt)
        P += (1000.0 * r) * dt  # scale factor so max q(~0.02) -> ~20 mm/day
        # Moisture budget with recycling and small diffusion
        q = q + E * dt - (1.0 - recycle_coeff) * r * dt
        q += kdiff * _lap(q) * dt
        # crude upwind advection relaxation
        qx = np.where(u >= 0, np.roll(q, 1, axis=1), np.roll(q, -1, axis=1))
        qy = np.where(v >= 0, np.roll(q, 1, axis=0), np.roll(q, -1, axis=0))
        q += 0.5 * (np.abs(u) / (wind_ref + 1e-6)) * (qx - q) * dt
        q += 0.5 * (np.abs(v) / (wind_ref + 1e-6)) * (qy - q) * dt
        q = np.clip(q, 0.0, qsat)
    P_full = _upsample_repeat(P.astype(np.float32), H, W, bs)
    q_full = _upsample_repeat(q.astype(np.float32), H, W, bs)
    return P_full, q_full


