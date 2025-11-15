"""Procedural planet viewer.

Generates a cached equirectangular elevation texture via 3D OpenSimplex fBM, then
renders either a lit globe (by sampling the texture using rotated surface normals)
or the flat map. Minimal UI with Tk interactivity for rotation and mode.
"""

from PIL import Image
import numpy as np
from opensimplex import OpenSimplex
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from temperature import generate_temperature_overlay, temperature_kelvin_for_lat
from atmosphere import generate_wind_field, render_wind_arrows, generate_precipitation
import logging
import time
from contextlib import contextmanager


_ELEV_TEX = None
_ELEV_SHAPE = (0, 0)
_ELEV_KEY = None  # (h, w, seed, octaves, freq, lac, gain)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("planetsim")

@contextmanager
def log_time(action: str):
    t0 = time.perf_counter()
    LOG.info(f"{action}...")
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        LOG.info(f"{action} done in {dt:.1f} ms")

# Lightweight caches for expensive view layers
_WIND_CACHE = {"key": None, "u": None, "v": None}
_PRECIP_CACHE = {"key": None, "P": None}

def invalidate_view_caches():
    _WIND_CACHE.update({"key": None, "u": None, "v": None})
    _PRECIP_CACHE.update({"key": None, "P": None})

# Simple multiprocessing tuning
_MP_ENABLED = True
_MP_WORKERS = None  # None → os.cpu_count()
_MP_CHUNK_ROWS = 32

# Settings persistence (terrain parameters)
SETTINGS_FILE = "settings.json"
_DEFAULT_SETTINGS = {
    "seed": 42,
    "octaves": 4,
    "freq": 1.2,
    "lac": 2.0,
    "gain": 0.5,
    "wind_arrows": 250,
    "wind_scale": 0.9,
}

def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            out = dict(_DEFAULT_SETTINGS)
            out.update({k: data.get(k, v) for k, v in _DEFAULT_SETTINGS.items()})
            return out
    except Exception:
        return dict(_DEFAULT_SETTINGS)

def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def _compute_elevation_block(r0: int, r1: int, tex_w: int, tex_h: int, seed: int, octaves: int, freq: float, lac: float, gain: float) -> tuple[int, np.ndarray]:
    """Worker: compute elevation rows [r0:r1) as float32 block.

    Each process creates its own noise generator (no shared mutable state) and
    computes a contiguous block of rows so results can be stitched in order.
    """
    noise = OpenSimplex(seed=seed)

    # Longitudes φ ∈ [-π, π), latitudes θ ∈ [-π/2, π/2]
    phi = np.linspace(-np.pi, np.pi, tex_w, endpoint=False)
    the = np.linspace(-np.pi / 2.0, np.pi / 2.0, tex_h)
    the_slice = the[r0:r1]
    PHI, THE = np.meshgrid(phi, the_slice)  # meshgrid over this block's latitudes
    cp, sp = np.cos(PHI), np.sin(PHI)
    ct, st = np.cos(THE), np.sin(THE)
    vx, vy, vz = (ct * cp).ravel(), st.ravel(), (ct * sp).ravel()  # flatten to 1D

    def fbm(a, b, c, octaves=octaves, freq=freq, lac=lac, gain=gain):
        a = float(a); b = float(b); c = float(c)
        amp = 1.0; f = freq; s = 0.0
        for _ in range(octaves): s += amp * noise.noise3(a * f, b * f, c * f); f *= lac; amp *= gain
        return 0.5 * (s + 1.0)

    total = vx.size
    block = np.fromiter((fbm(vx[i], vy[i], vz[i]) for i in range(total)), dtype=np.float32, count=total)
    return r0, block.reshape(r1 - r0, tex_w)

def ensure_elevation(size: int, seed: int = 42, octaves: int = 4, freq: float = 1.2, lac: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """Build and cache an equirectangular elevation map once. Returns float in [0,1].

    - Texture shape: height=size latitudes, width=2*size longitudes.
    - Each texel samples 3D fBM at the unit sphere direction for that lon/lat.
    """
    global _ELEV_TEX, _ELEV_SHAPE, _ELEV_KEY
    
    # If a loaded heightmap is in the cache, use it (don't regenerate)
    if _ELEV_TEX is not None and _ELEV_KEY is not None and isinstance(_ELEV_KEY, tuple) and len(_ELEV_KEY) >= 1:
        if _ELEV_KEY[0] == "loaded":
            # Loaded heightmap is cached, return it
            return _ELEV_TEX
    
    tex_w, tex_h = size * 2, size  # equirectangular: W=360°, H=180°
    key = (size, int(seed), int(octaves), float(freq), float(lac), float(gain))
    if _ELEV_TEX is not None and _ELEV_KEY == key:
        return _ELEV_TEX

    # Parallel path: split rows across processes
    workers = (_MP_WORKERS or os.cpu_count() or 1) if _MP_ENABLED else 1
    if workers > 1:
        elev = np.empty((tex_h, tex_w), dtype=np.float32)  # destination texture
        with log_time(f"Generating elevation in {workers} processes"):
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = []
                for r0 in range(0, tex_h, _MP_CHUNK_ROWS):  # submit row blocks
                    r1 = min(r0 + _MP_CHUNK_ROWS, tex_h)
                    futures.append(pool.submit(_compute_elevation_block, r0, r1, tex_w, tex_h, seed, int(octaves), float(freq), float(lac), float(gain)))
                for fut in as_completed(futures):
                    r0, block = fut.result()
                    elev[r0:r0 + block.shape[0], :] = block  # place rows back into texture
        _ELEV_TEX = elev
    else:
        noise = OpenSimplex(seed=seed)
        # Longitudes φ ∈ [-π, π), latitudes θ ∈ [-π/2, π/2]
        phi = np.linspace(-np.pi, np.pi, tex_w, endpoint=False)
        the = np.linspace(-np.pi / 2.0, np.pi / 2.0, tex_h)
        PHI, THE = np.meshgrid(phi, the)
        cp, sp = np.cos(PHI), np.sin(PHI)
        ct, st = np.cos(THE), np.sin(THE)
        # Unit direction vectors for each (φ, θ)
        vx, vy, vz = (ct * cp).ravel(), st.ravel(), (ct * sp).ravel()

        # Small fBM helper for soft multi-scale features; output in [0,1]
        def fbm(a, b, c, octaves=octaves, freq=freq, lac=lac, gain=gain):
            a = float(a); b = float(b); c = float(c)
            amp = 1.0; f = freq; s = 0.0
            for _ in range(octaves): s += amp * noise.noise3(a * f, b * f, c * f); f *= lac; amp *= gain
            return 0.5 * (s + 1.0)

        total = vx.size
        with log_time("Generating elevation (single process)"):
            elev = np.fromiter((fbm(vx[i], vy[i], vz[i]) for i in range(total)), dtype=np.float32, count=total)  # 1D
        _ELEV_TEX = elev.reshape(tex_h, tex_w)
    _ELEV_SHAPE = (tex_h, tex_w)
    _ELEV_KEY = key
    return _ELEV_TEX


def colorize(elev: np.ndarray) -> np.ndarray:
    """Map elevation in [0,1] to RGB using a piecewise-linear palette.

    Breakpoints span ocean→beach→land→snow; linear interpolation between stops.
    Raven Maps style: muted earth tones, dense breakpoints for subtle gradients.
    """
    # Raven Maps inspired gradient: 20 breakpoints optimized for terrain detail
    # After normalization: 0.0 = ocean, 0.0-0.03 = low (0-100m), 0.03-1.0 = higher (100-8848m)
    bp = np.array([
        0.0,      # Ocean
        0.000001, # Immediate coast
        0.006,    # ~20m: deltas, tidal flats
        0.012,    # ~40m: river valleys
        0.018,    # ~60m: coastal plains
        0.025,    # ~80m: low plains
        0.035,    # ~120m: plains
        0.055,    # ~230m: interior lowlands
        0.085,    # ~350m: rolling plains
        0.125,    # ~470m: elevated plains
        0.175,    # ~770m: low hills
        0.240,    # ~1200m: hills
        0.320,    # ~1700m: high hills
        0.420,    # ~2400m: foothills
        0.530,    # ~3300m: low mountains
        0.640,    # ~4400m: mountains
        0.740,    # ~5500m: high mountains
        0.850,    # ~7000m: alpine peaks
        0.950,    # ~8200m: snow peaks
        1.000,    # ~8848m: highest peaks
    ], dtype=np.float32)
    
    # Muted, desaturated earth tones for natural appearance
    col = np.array([
        [0.02, 0.08, 0.22],   # Ocean: deep blue
        [0.68, 0.72, 0.58],   # Coast: sandy tan
        [0.64, 0.69, 0.54],   # Deltas: light tan-green
        [0.60, 0.66, 0.50],   # River valleys: tan-green
        [0.56, 0.63, 0.47],   # Coastal plains: muted green-tan
        [0.53, 0.60, 0.45],   # Low plains: olive-tan
        [0.51, 0.58, 0.43],   # Plains: olive
        [0.52, 0.57, 0.43],   # Interior lowlands: yellow-olive
        [0.54, 0.58, 0.44],   # Rolling plains: light olive
        [0.56, 0.58, 0.45],   # Elevated plains: tan-olive
        [0.58, 0.58, 0.46],   # Low hills: yellow-tan
        [0.61, 0.58, 0.47],   # Hills: tan
        [0.63, 0.57, 0.47],   # High hills: tan-brown
        [0.66, 0.56, 0.46],   # Foothills: brown-tan
        [0.68, 0.54, 0.45],   # Low mountains: brown
        [0.71, 0.54, 0.44],   # Mountains: darker brown
        [0.74, 0.57, 0.47],   # High mountains: brown-gray
        [0.80, 0.70, 0.62],   # Alpine peaks: tan-gray
        [0.90, 0.88, 0.86],   # Snow peaks: light gray
        [1.00, 1.00, 1.00],   # Highest peaks: pure white
    ], dtype=np.float32)
    
    # Calculate interpolated colors
    i = np.clip(np.searchsorted(bp, elev, side="right") - 1, 0, len(bp) - 2)
    c0, c1 = col[i], col[i + 1]
    t = (elev - bp[i]) / (bp[i + 1] - bp[i] + 1e-9)
    result = c0 + (c1 - c0) * t[..., None]
    
    # Force pixels at exactly 0.0 to be pure ocean color (no interpolation)
    ocean_mask = (elev == 0.0)
    result[ocean_mask] = col[0]
    
    return result


def generate_sphere_image(size: int = 512, radius: float = 0.9, rot=(0.0, 0.0, 0.0), *, view: str = "Terrain", seed: int = 42, octaves: int = 4, freq: float = 1.2, lac: float = 2.0, gain: float = 0.5, day_of_year: int = 1) -> Image.Image:
    """Render a fully lit sphere by sampling the cached terrain. radius<1 zooms out.

    - Build a canvas-space unit disk and reconstruct Z for the sphere surface.
    - Rotate normals by yaw/pitch/roll, map to (φ, θ), then sample the texture.
    - No shading; colors come solely from elevation, with optional overlays.
    """
    lin = np.linspace(-1.0, 1.0, size)
    u, v = np.meshgrid(lin, -lin)  # invert Y so lighting feels natural
    r2_canvas = u * u + v * v
    mask = r2_canvas <= (radius * radius)  # boolean disk mask

    r2_unit = r2_canvas / (radius * radius)  # normalized squared radius inside the disk
    z = np.zeros_like(u)
    z[mask] = np.sqrt(1.0 - r2_unit[mask])

    x = u / radius
    y = v / radius
    normals = np.stack((x, y, z), axis=-1)  # unrotated surface normals
    norms = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
    n0 = normals / norms

    # rotation (yaw, pitch, roll) → Rz * Ry * Rx
    yaw, pitch, roll = rot
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(roll), np.sin(roll)
    Ry = np.array([[ cy, 0.0,  sy], [0.0, 1.0, 0.0], [-sy, 0.0,  cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0,  cx, -sx], [0.0,  sx,  cx]])
    Rz = np.array([[ cz, -sz, 0.0], [ sz,  cz, 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ Ry @ Rx
    n = n0 @ R.T

    # Fully lit: sample cached elevation and map to color; optionally blend temperature
    tex = ensure_elevation(size, seed=seed, octaves=octaves, freq=freq, lac=lac, gain=gain)
    
    # Check if we have a loaded heightmap and flip it for correct globe orientation
    _, elev_key = get_elevation_cache()
    if elev_key is not None and isinstance(elev_key, tuple) and len(elev_key) >= 1 and elev_key[0] == "loaded":
        # Flip loaded heightmap horizontally for globe view only
        tex = np.fliplr(tex)
    
    tex_h, tex_w = tex.shape
    # normal → spherical → texture coords (equirectangular mapping)
    phi = np.arctan2(n[..., 2], n[..., 0])
    theta = np.arcsin(np.clip(n[..., 1], -1.0, 1.0))
    uu = (phi + np.pi) / (2.0 * np.pi)
    vv = 0.5 - (theta / np.pi)
    ix = np.clip((uu * (tex_w - 1)).astype(np.int32), 0, tex_w - 1)
    iy = np.clip((vv * (tex_h - 1)).astype(np.int32), 0, tex_h - 1)
    elev_img = np.zeros_like(u, dtype=np.float32)
    idx = np.where(mask)
    elev_img[idx] = tex[iy[idx], ix[idx]]
    rgbf = colorize(elev_img)
    if view == "Temperature":
        overlay_tex = generate_temperature_overlay(tex_h, tex_w, day_of_year=day_of_year, elevation=tex)
        overlay_img = np.zeros((*elev_img.shape, 3), dtype=np.float32)
        overlay_img[idx] = overlay_tex[iy[idx], ix[idx], :]
        alpha = 0.5
        rgbf = (1.0 - alpha) * rgbf + alpha * overlay_img
    elif view == "Wind":
        # Overlay wind arrows (project equirectangular arrows via sampling)
        from atmosphere import generate_wind_field, render_wind_arrows
        u, v = generate_wind_field(tex_h, tex_w, elevation=tex)
        arrows = render_wind_arrows(tex_h, tex_w, u, v, target_arrows=250)
        arr_img = np.zeros((*elev_img.shape, 3), dtype=np.float32)
        arr_img[idx] = arrows[iy[idx], ix[idx], :]
        rgbf = np.clip(rgbf + arr_img, 0.0, 1.0)
    rgb = (np.clip(rgbf, 0.0, 1.0) * 255).astype(np.uint8)
    rgb[~mask] = 0

    return Image.fromarray(rgb)


def get_elevation_cache() -> tuple[np.ndarray | None, tuple | None]:
    """Get current elevation cache state."""
    return _ELEV_TEX, _ELEV_KEY


def clear_elevation_cache() -> None:
    """Clear elevation cache."""
    global _ELEV_TEX, _ELEV_SHAPE, _ELEV_KEY
    _ELEV_TEX = None
    _ELEV_SHAPE = (0, 0)
    _ELEV_KEY = None


def set_elevation_cache(elevation: np.ndarray, key: tuple | None = None) -> None:
    """Set elevation cache with custom data (e.g., loaded heightmap).
    
    Args:
        elevation: (H, W) elevation array in [0, 1]
        key: Optional cache key (use special key for loaded data)
    """
    global _ELEV_TEX, _ELEV_SHAPE, _ELEV_KEY
    _ELEV_TEX = elevation.astype(np.float32)
    _ELEV_SHAPE = elevation.shape
    _ELEV_KEY = key if key is not None else ("loaded", elevation.shape)


