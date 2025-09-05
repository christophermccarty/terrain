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


_ELEV_TEX = None
_ELEV_SHAPE = (0, 0)

# Simple multiprocessing tuning
_MP_ENABLED = True
_MP_WORKERS = None  # None → os.cpu_count()
_MP_CHUNK_ROWS = 32


def _compute_elevation_block(r0: int, r1: int, tex_w: int, tex_h: int, seed: int) -> tuple[int, np.ndarray]:
    """Worker: compute elevation rows [r0:r1) as float32 block."""
    noise = OpenSimplex(seed=seed)

    # Longitudes φ ∈ [-π, π), latitudes θ ∈ [-π/2, π/2]
    phi = np.linspace(-np.pi, np.pi, tex_w, endpoint=False)
    the = np.linspace(-np.pi / 2.0, np.pi / 2.0, tex_h)
    the_slice = the[r0:r1]
    PHI, THE = np.meshgrid(phi, the_slice)
    cp, sp = np.cos(PHI), np.sin(PHI)
    ct, st = np.cos(THE), np.sin(THE)
    vx, vy, vz = (ct * cp).ravel(), st.ravel(), (ct * sp).ravel()

    def fbm(a, b, c, octaves=4, freq=1.2, lac=2.0, gain=0.5):
        a = float(a); b = float(b); c = float(c)
        amp = 1.0; f = freq; s = 0.0
        for _ in range(octaves): s += amp * noise.noise3(a * f, b * f, c * f); f *= lac; amp *= gain
        return 0.5 * (s + 1.0)

    total = vx.size
    block = np.fromiter((fbm(vx[i], vy[i], vz[i]) for i in range(total)), dtype=np.float32, count=total)
    return r0, block.reshape(r1 - r0, tex_w)

def _ensure_elevation(size: int, seed: int = 42) -> np.ndarray:
    """Build and cache an equirectangular elevation map once. Returns float in [0,1].

    - Texture shape: height=size latitudes, width=2*size longitudes.
    - Each texel samples 3D fBM at the unit sphere direction for that lon/lat.
    """
    global _ELEV_TEX, _ELEV_SHAPE
    tex_w, tex_h = size * 2, size  # equirectangular: W=360°, H=180°
    if _ELEV_TEX is not None and _ELEV_SHAPE == (tex_h, tex_w):
        return _ELEV_TEX

    # Parallel path: split rows across processes
    workers = (_MP_WORKERS or os.cpu_count() or 1) if _MP_ENABLED else 1
    if workers > 1:
        elev = np.empty((tex_h, tex_w), dtype=np.float32)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = []
            for r0 in range(0, tex_h, _MP_CHUNK_ROWS):
                r1 = min(r0 + _MP_CHUNK_ROWS, tex_h)
                futures.append(pool.submit(_compute_elevation_block, r0, r1, tex_w, tex_h, seed))
            for fut in as_completed(futures):
                r0, block = fut.result()
                elev[r0:r0 + block.shape[0], :] = block
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
        def fbm(a, b, c, octaves=4, freq=1.2, lac=2.0, gain=0.5):
            a = float(a); b = float(b); c = float(c)
            amp = 1.0; f = freq; s = 0.0
            for _ in range(octaves): s += amp * noise.noise3(a * f, b * f, c * f); f *= lac; amp *= gain
            return 0.5 * (s + 1.0)

        total = vx.size
        elev = np.fromiter((fbm(vx[i], vy[i], vz[i]) for i in range(total)), dtype=np.float32, count=total)
        _ELEV_TEX = elev.reshape(tex_h, tex_w)
    _ELEV_SHAPE = (tex_h, tex_w)
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


def generate_sphere_image(size: int = 512, radius: float = 0.9, rot=(0.0, 0.0, 0.0)) -> Image.Image:
    """Render a fully lit sphere by sampling the cached terrain. radius<1 zooms out.

    - Build a canvas-space unit disk and reconstruct Z for the sphere surface.
    - Rotate normals by yaw/pitch/roll, map to (φ, θ), then sample the texture.
    - No shading; colors come solely from elevation.
    """
    lin = np.linspace(-1.0, 1.0, size)
    u, v = np.meshgrid(lin, -lin)  # invert Y so lighting feels natural
    r2_canvas = u * u + v * v
    mask = r2_canvas <= (radius * radius)

    r2_unit = r2_canvas / (radius * radius)
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
    tex = _ensure_elevation(size)
    tex_h, tex_w = tex.shape
    # normal → spherical → texture coords
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

    root = tk.Tk()
    root.title(f"Sphere {size}x{size} (262,144 cells)")
    root.resizable(False, False)

    # Controls
    mode_var = tk.StringVar(value="globe")
    controls = tk.Frame(root)
    controls.pack(fill="x")
    tk.Radiobutton(controls, text="Globe", variable=mode_var, value="globe").pack(side="left")
    tk.Radiobutton(controls, text="Map", variable=mode_var, value="map").pack(side="left")

    # Profile initial generation once at startup
    profiler = cProfile.Profile()
    profiler.enable()
    img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll))
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)
    tk_img = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(root, width=size, height=size, highlightthickness=0)
    canvas.pack()
    img_id = canvas.create_image(0, 0, image=tk_img, anchor="nw")

    def render():
        nonlocal tk_img
        if mode_var.get() == "globe":
            new_img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll))
            canvas.config(width=size, height=size)
        else:
            tex = _ensure_elevation(size)
            rgbf = _colorize(tex)
            arr = (np.clip(rgbf, 0.0, 1.0) * 255).astype(np.uint8)
            new_img = Image.fromarray(arr)
            h, w = tex.shape
            canvas.config(width=w, height=h)
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)

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
    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    main()


