"""Procedural planet viewer.

Generates a cached equirectangular elevation texture via 3D OpenSimplex fBM, then
renders either a lit globe (by sampling the texture using rotated surface normals)
or the flat map. Minimal UI with Tk interactivity for rotation and mode.
"""

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from opensimplex import OpenSimplex
import cProfile
import pstats
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from temperature import generate_temperature_overlay, temperature_kelvin_for_lat


_ELEV_TEX = None
_ELEV_SHAPE = (0, 0)
_ELEV_KEY = None  # (h, w, seed, octaves, freq, lac, gain)

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
}

def _load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            out = dict(_DEFAULT_SETTINGS)
            out.update({k: data.get(k, v) for k, v in _DEFAULT_SETTINGS.items()})
            return out
    except Exception:
        return dict(_DEFAULT_SETTINGS)

def _save_settings(settings: dict) -> None:
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

def _ensure_elevation(size: int, seed: int = 42, octaves: int = 4, freq: float = 1.2, lac: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """Build and cache an equirectangular elevation map once. Returns float in [0,1].

    - Texture shape: height=size latitudes, width=2*size longitudes.
    - Each texel samples 3D fBM at the unit sphere direction for that lon/lat.
    """
    global _ELEV_TEX, _ELEV_SHAPE, _ELEV_KEY
    tex_w, tex_h = size * 2, size  # equirectangular: W=360°, H=180°
    key = (tex_h, tex_w, int(seed), int(octaves), float(freq), float(lac), float(gain))
    if _ELEV_TEX is not None and _ELEV_KEY == key:
        return _ELEV_TEX

    # Parallel path: split rows across processes
    workers = (_MP_WORKERS or os.cpu_count() or 1) if _MP_ENABLED else 1
    if workers > 1:
        elev = np.empty((tex_h, tex_w), dtype=np.float32)  # destination texture
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
        elev = np.fromiter((fbm(vx[i], vy[i], vz[i]) for i in range(total)), dtype=np.float32, count=total)  # 1D
        _ELEV_TEX = elev.reshape(tex_h, tex_w)
    _ELEV_SHAPE = (tex_h, tex_w)
    _ELEV_KEY = key
    return _ELEV_TEX


def _colorize(elev: np.ndarray) -> np.ndarray:
    """Map elevation in [0,1] to RGB using a piecewise-linear palette.

    Breakpoints span ocean→beach→land→snow; linear interpolation between stops.
    """
    bp = np.array([0.0, 0.35, 0.48, 0.55, 0.7, 0.85, 1.0], dtype=np.float32)
    col = np.array([
        [0.02, 0.08, 0.20],   # deep ocean
        [0.00, 0.40, 0.70],   # shallow ocean
        [0.94, 0.86, 0.62],   # beach
        [0.10, 0.60, 0.20],   # lowland
        [0.45, 0.34, 0.22],   # highland
        [0.80, 0.80, 0.80],   # snow line
        [1.00, 1.00, 1.00],   # peaks
    ], dtype=np.float32)
    i = np.clip(np.searchsorted(bp, elev, side="right") - 1, 0, len(bp) - 2)
    c0, c1 = col[i], col[i + 1]
    t = (elev - bp[i]) / (bp[i + 1] - bp[i] + 1e-9)
    return c0 + (c1 - c0) * t[..., None]


def generate_sphere_image(size: int = 512, radius: float = 0.9, rot=(0.0, 0.0, 0.0), *, seed: int = 42, octaves: int = 4, freq: float = 1.2, lac: float = 2.0, gain: float = 0.5) -> Image.Image:
    """Render a fully lit sphere by sampling the cached terrain. radius<1 zooms out.

    - Build a canvas-space unit disk and reconstruct Z for the sphere surface.
    - Rotate normals by yaw/pitch/roll, map to (φ, θ), then sample the texture.
    - No shading; colors come solely from elevation.
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

    # Fully lit: sample cached elevation and map to color, no shading
    tex = _ensure_elevation(size, seed=seed, octaves=octaves, freq=freq, lac=lac, gain=gain)
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
    rgbf = _colorize(elev_img)
    rgb = (np.clip(rgbf, 0.0, 1.0) * 255).astype(np.uint8)
    rgb[~mask] = 0

    return Image.fromarray(rgb)


def main() -> None:
    """Tiny Tk UI to toggle globe/map and rotate with keys.

    Keys: arrows=yaw/pitch, A/D=roll, R=reset, Esc=quit. Radio: globe vs map.
    """
    # 262,144 cells = 512 x 512
    size = 512
    yaw = 0.0; pitch = 0.0; roll = 0.0
    settings = _load_settings()

    root = tk.Tk()
    root.title(f"Sphere {size}x{size} (262,144 cells)")
    root.resizable(False, False)

    # Controls
    mode_var = tk.StringVar(value="map")
    view_var = tk.StringVar(value="Terrain")
    latlon_var = tk.StringVar(value="")
    controls = tk.Frame(root)
    controls.pack(fill="x")
    tk.Radiobutton(controls, text="Globe", variable=mode_var, value="globe").pack(side="left")
    tk.Radiobutton(controls, text="Map", variable=mode_var, value="map").pack(side="left")
    tk.Label(controls, text="View").pack(side="left", padx=(8,0))
    tk.OptionMenu(controls, view_var, "Terrain", "Temperature").pack(side="left")

    # Terrain parameter inputs
    seed_var = tk.IntVar(value=int(settings["seed"]))
    octaves_var = tk.IntVar(value=int(settings["octaves"]))
    freq_var = tk.DoubleVar(value=float(settings["freq"]))
    lac_var = tk.DoubleVar(value=float(settings["lac"]))
    gain_var = tk.DoubleVar(value=float(settings["gain"]))

    def add_labeled_entry(parent, label, var, width=6):
        frm = tk.Frame(parent)
        frm.pack(side="left", padx=4)
        tk.Label(frm, text=label).pack(side="left")
        tk.Entry(frm, textvariable=var, width=width).pack(side="left")
        return frm

    add_labeled_entry(controls, "Seed", seed_var)
    add_labeled_entry(controls, "Oct", octaves_var)
    add_labeled_entry(controls, "Freq", freq_var)
    add_labeled_entry(controls, "Lac", lac_var)
    add_labeled_entry(controls, "Gain", gain_var)

    def do_regen():
        nonlocal tk_img
        # Clear cache so new params take effect next call
        global _ELEV_TEX, _ELEV_SHAPE, _ELEV_KEY
        _ELEV_TEX = None; _ELEV_SHAPE = (0, 0); _ELEV_KEY = None
        p = {
            "seed": int(seed_var.get()),
            "octaves": int(octaves_var.get()),
            "freq": float(freq_var.get()),
            "lac": float(lac_var.get()),
            "gain": float(gain_var.get()),
        }
        if mode_var.get() == "globe":
            new_img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll), **p)
            canvas.config(width=size, height=size)
        else:
            tex = _ensure_elevation(size, **p)
            rgbf = _colorize(tex)
            arr = (np.clip(rgbf, 0.0, 1.0) * 255).astype(np.uint8)
            new_img = Image.fromarray(arr)
            h, w = tex.shape
            canvas.config(width=w, height=h)
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)

    tk.Button(controls, text="Regenerate", command=do_regen).pack(side="left", padx=6)
    latlon_label = tk.Label(controls, textvariable=latlon_var)
    latlon_label.pack(side="right")

    # Profile initial generation once at startup
    profiler = cProfile.Profile()
    profiler.enable()
    # Default: Map view
    tex0 = _ensure_elevation(size, seed=settings["seed"], octaves=settings["octaves"], freq=settings["freq"], lac=settings["lac"], gain=settings["gain"])
    rgbf0 = _colorize(tex0)
    arr0 = (np.clip(rgbf0, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(arr0)
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)
    tk_img = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(root, width=tex0.shape[1], height=tex0.shape[0], highlightthickness=0)
    canvas.pack()
    img_id = canvas.create_image(0, 0, image=tk_img, anchor="nw")

    def render():
        nonlocal tk_img
        if mode_var.get() == "globe":
            new_img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll), seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
            canvas.config(width=size, height=size)
        else:
            tex = _ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
            base_rgb = _colorize(tex)
            if view_var.get() == "Temperature":
                h, w = tex.shape
                overlay = generate_temperature_overlay(h, w)
                alpha = 0.7 # Sets temperature map opacity
                comb = (1.0 - alpha) * base_rgb + alpha * overlay
                arr = (np.clip(comb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                arr = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            new_img = Image.fromarray(arr)
            h, w = tex.shape
            canvas.config(width=w, height=h)
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)
        # Clear lat/lon when switching out of map
        if mode_var.get() != "map":
            latlon_var.set("")

    def on_motion(e):
        # Only show in Map mode; map image size equals texture size (H,W)
        if mode_var.get() != "map":
            return
        tex = _ensure_elevation(size)
        h, w = tex.shape
        x, y = e.x, e.y
        if 0 <= x < w and 0 <= y < h:
            # Equirectangular: x∈[0,w) -> φ∈[-π,π), y∈[0,h) -> θ∈[-π/2,π/2]
            lon = (x / w) * 360.0 - 180.0
            lat = 90.0 - (y / h) * 180.0
            # Compute temperature at this latitude
            T = temperature_kelvin_for_lat(np.deg2rad(lat))
            latlon_var.set(f"lat {lat:.2f}°, lon {lon:.2f}°, T {T:.1f} K")
        else:
            latlon_var.set("")

    step = np.deg2rad(5.0)
    def on_key(e):
        nonlocal yaw, pitch, roll
        if e.keysym == "Left":
            yaw -= step
        elif e.keysym == "Right":
            yaw += step
        elif e.keysym == "Up":
            pitch -= step
        elif e.keysym == "Down":
            pitch += step
        elif e.keysym.lower() == "a":
            roll -= step
        elif e.keysym.lower() == "d":
            roll += step
        elif e.keysym.lower() == "r":
            yaw = pitch = roll = 0.0
        else:
            return
        if mode_var.get() == "globe":
            render()

    root.bind("<Key>", on_key)
    mode_var.trace_add("write", lambda *_: render())
    view_var.trace_add("write", lambda *_: render())
    def on_close():
        # Persist current UI parameters to settings.json
        s = {
            "seed": int(seed_var.get()),
            "octaves": int(octaves_var.get()),
            "freq": float(freq_var.get()),
            "lac": float(lac_var.get()),
            "gain": float(gain_var.get()),
        }
        _save_settings(s)
        root.destroy()

    root.bind("<Escape>", lambda e: on_close())
    root.protocol("WM_DELETE_WINDOW", on_close)
    canvas.bind("<Motion>", on_motion)
    root.mainloop()


if __name__ == "__main__":
    main()


