"""Main entry point for planet simulator.

Launches the GUI application for viewing and interacting with the planet simulation.
All modules are kept separated: terrain, atmosphere, temperature, and simulate.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cProfile
import pstats
import logging
import time
from pathlib import Path
from threading import Thread, Event
from queue import Queue, Empty

from terrain import (
    generate_sphere_image,
    project_equirect_on_globe,
    ensure_elevation,
    colorize,
    load_settings,
    save_settings,
    invalidate_view_caches,
    log_time,
    get_elevation_cache,
    clear_elevation_cache,
    set_elevation_cache,
    LOG,
)
from atmosphere import generate_wind_field, render_wind_arrows, wind_speed_to_rgb, generate_precipitation
from temperature import generate_temperature_overlay, temperature_kelvin_for_lat
from ocean import generate_ocean_currents
from masks import get_masks
from terrain import precipitation_to_rgb, cloud_cover_to_rgb
from simulate import PlanetState, create_initial_state, simulate_step, simulate_multiple_steps, save_state, load_state, TimeScaleMode
from diagnostics import ClimateDiagnostics
import graphs

AUTOSAVE_PATH = Path("saves/autosave.pkl")

# Lightweight caches for expensive view layers
_WIND_CACHE = {"key": None, "u": None, "v": None}
_OCEAN_CURRENT_CACHE = {"key": None, "u": None, "v": None}
_PRECIP_VIEW_CACHE = {"key": None, "P": None}


class SimulationThread(Thread):
    """Background thread that runs physics simulation independently of UI."""

    def __init__(self, initial_state, days_per_step=1.0, wind_block_size=8, diagnostics=None,
                 time_scale_mode: TimeScaleMode = TimeScaleMode.DAILY):
        super().__init__(daemon=True)
        self.state = initial_state
        self.days_per_step = days_per_step
        self.wind_block_size = wind_block_size
        self.diagnostics = diagnostics
        self.time_scale_mode = time_scale_mode
        self.running = Event()
        self.paused = Event()
        self.paused.set()  # Start paused
        self.state_queue = Queue(maxsize=1)  # Only keep latest state
        self.component_queue = Queue(maxsize=1)  # Track temperature components

    def run(self):
        """Main simulation loop (runs in background thread)."""
        self.running.set()
        while self.running.is_set():
            if self.paused.is_set():
                time.sleep(0.05)  # Sleep when paused (50ms)
                continue

            try:
                # Select sub-stepping strategy based on time scale mode.
                # Each entry is (step_days, update_wind):
                #   DAILY   — 1 × 1-day full physics (most accurate)
                #   WEEKLY  — 7 × 1-day full physics (7 simulated days per frame)
                #   MONTHLY — 5 × 6-day steps, no wind (≈30 days; ~5× faster than 30 daily)
                #   ANNUAL  — 52 × 7-day steps, no wind (≈364 days; stable large-step physics)
                mode = self.time_scale_mode
                if mode == TimeScaleMode.WEEKLY:
                    substeps = [(1.0, True)] * 7
                elif mode == TimeScaleMode.MONTHLY:
                    substeps = [(6.0, False)] * 5
                elif mode == TimeScaleMode.ANNUAL:
                    substeps = [(7.0, False)] * 52
                else:  # DAILY (default)
                    substeps = [(1.0, True)]

                new_state = self.state
                temp_components: dict = {}
                for step_days, do_wind in substeps:
                    new_state, temp_components = simulate_step(
                        new_state,
                        days=step_days,
                        wind_block_size=self.wind_block_size,
                        update_wind=do_wind,
                        debug_log=False,
                        track_components=self.diagnostics is not None,
                        time_scale=mode,
                    )

                    # Record diagnostics each sub-step for correct time-averaging
                    if self.diagnostics is not None:
                        self.diagnostics.record_step(
                            new_state,
                            new_state.day_of_year,
                            days_elapsed=step_days,
                            component_contributions=temp_components
                        )

                self.state = new_state

                # Push final state to UI (non-blocking, drop if UI busy)
                try:
                    self.state_queue.put_nowait(new_state)
                    self.component_queue.put_nowait(temp_components)
                except Exception:
                    pass  # Drop frame if queue full

            except Exception as e:
                LOG.error(f"Simulation thread error: {e}")
                self.paused.set()  # Auto-pause on error

    def pause(self):
        """Pause simulation."""
        self.paused.set()

    def resume(self):
        """Resume simulation."""
        self.paused.clear()

    def stop(self):
        """Stop simulation thread."""
        self.running.clear()

    def update_days_per_step(self, days):
        """Update simulation speed (legacy; kept for backward compatibility)."""
        self.days_per_step = days

    def update_time_scale(self, mode: TimeScaleMode):
        """Switch time-scale mode (affects sub-stepping strategy)."""
        self.time_scale_mode = mode

    def update_wind_block_size(self, block_size):
        """Update wind resolution."""
        self.wind_block_size = block_size


def main() -> None:
    """Tiny Tk UI to toggle globe/map and rotate with keys.

    Keys: arrows=yaw/pitch, A/D=roll, R=reset, Esc=quit. Radio: globe vs map.
    """
    # 262,144 cells = 512 x 512
    size = 512
    yaw = 0.0; pitch = 0.0; roll = 0.0
    settings = load_settings()
    _auto_save_enabled: bool = bool(settings.get("auto_save_state", False))
    _saved_state_days: float | None = None  # total_days of the loaded autosave (for display)
    default_settings = {
        "seed": 42,
        "octaves": 4,
        "freq": 1.2,
        "lac": 2.0,
        "gain": 0.5,
        "wind_arrows": 250,
        "wind_scale": 0.9,
        # Wind model resolution decoupled from temp/precip resolution.
        # Larger => fewer wind cells => faster, but more approximate.
        "wind_block_size": 8,
    }

    root = tk.Tk()
    root.title(f"Sphere {size}x{size} (262,144 cells)")
    root.resizable(True, True)

    # Simulation state
    sim_state: PlanetState | None = None
    # Currently active save file (File>Open / File>Save As changes this).
    # Restored from the last session so the app picks up the same planet
    # it was last working on; falls back to the legacy autosave path.
    _current_state_path: Path = Path(settings.get("last_state_path", str(AUTOSAVE_PATH)))
    sim_thread: SimulationThread | None = None
    sim_running = False
    sim_paused = False
    sim_speed = 1.0  # days per step
    last_mouse_pos = (0, 0)  # Track last mouse position for cursor updates
    display_scale_x: float = 1.0  # display pixels per sim pixel (x)
    display_scale_y: float = 1.0  # display pixels per sim pixel (y)
    _resize_job = None  # debounce id for window Configure events
    _sim_ever_started = False  # False until user clicks Start for the first time
    _last_render_arr: np.ndarray | None = None  # latest rendered map pixels for zoom preview

    # Terrain mode: "procedural" or "loaded"
    terrain_mode = "procedural"
    loaded_heightmap_path = None
    
    # --- Main layout scaffold ---
    # A persistent bottom status bar (packed first so it keeps its slice of
    # the window), a resizable horizontal split between a tabbed sidebar
    # (Terrain / Simulation, the latter also holding view-mode/type controls)
    # and the map/globe area. The menu bar itself is assembled near the end
    # of main(), once every command callback it references has been defined.
    status_bar = ttk.Frame(root, relief="sunken", padding=(4, 2))
    status_bar.pack(side="bottom", fill="x")

    body = ttk.Panedwindow(root, orient="horizontal")
    body.pack(side="top", fill="both", expand=True)

    sidebar = ttk.Frame(body)
    body.add(sidebar, weight=0)

    notebook = ttk.Notebook(sidebar)
    notebook.pack(fill="both", expand=True)

    terrain_tab = ttk.Frame(notebook, padding=6)
    sim_tab = ttk.Frame(notebook, padding=6)
    notebook.add(sim_tab, text="Simulation")
    notebook.add(terrain_tab, text="Terrain")
    notebook.select(sim_tab)

    # Controls
    mode_var = tk.StringVar(value="map")
    view_var = tk.StringVar(value="Terrain")
    latlon_var = tk.StringVar(value="")

    mode_row = ttk.Frame(sim_tab)
    mode_row.pack(fill="x", pady=2)
    ttk.Radiobutton(mode_row, text="Globe", variable=mode_var, value="globe").pack(side="left")
    ttk.Radiobutton(mode_row, text="Map", variable=mode_var, value="map").pack(side="left")

    view_row = ttk.Frame(sim_tab)
    view_row.pack(fill="x", pady=2)
    ttk.Label(view_row, text="View:").pack(side="left")
    view_combo = ttk.Combobox(
        view_row,
        textvariable=view_var,
        state="readonly",
        width=16,
        values=(
            "Terrain",
            "Temperature",
            "Ocean Temperature",
            "Precipitation",
            "Biomes",
            "Wind Arrows",
            "Ocean Currents",
            "Wind Particles",
            "Cloud Cover",
        ),
    )
    view_combo.pack(side="left", padx=(4, 0))

    # Diagnostics
    diagnostics = ClimateDiagnostics(track_history=True)
    graphs_enabled_var = tk.BooleanVar(value=False)
    graphs_controller = graphs.GraphsController(
        root,
        get_state=lambda: sim_state,
        diagnostics=diagnostics,
        history_days=365.0,
        update_ms=1000,
        toggle_var=graphs_enabled_var,
    )

    # --- Wind particle visualization state (map mode) ---
    particle_xy: np.ndarray | None = None  # (N,2) float32, x/y in pixel space
    particle_age: np.ndarray | None = None  # (N,) int32
    trail: np.ndarray | None = None  # (H,W) float32 intensity
    base_wind_rgb_u8: np.ndarray | None = None  # (H,W,3) uint8
    last_wind_key = None
    last_anim_t = 0.0
    particle_anim_running = False
    last_particle_mode: str | None = None

    # --- Ocean current particle visualization state ---
    oc_particle_xy: np.ndarray | None = None
    oc_particle_age: np.ndarray | None = None
    oc_trail: np.ndarray | None = None
    oc_base_rgb_u8: np.ndarray | None = None
    oc_last_uv_key = None
    oc_last_anim_t = 0.0
    oc_anim_running = False
    oc_last_particle_mode: str | None = None

    def _particle_count_scale() -> float:
        """Scale decorative particle count down at coarser time-scale modes.

        Wind/ocean particles are purely decorative — animation smoothness
        matters less when each simulated step represents a week/month/year
        rather than a day, so it isn't worth paying full per-frame
        update/render cost at those modes (PLAN.md Phase 5).
        """
        return {"1 Day": 1.0, "1 Week": 1.0, "1 Month": 0.5, "1 Year": 0.25}.get(
            time_scale_var.get(), 1.0
        )

    def _init_particles(h: int, w: int, n: int) -> None:
        """Initialize particles and trail buffer for wind particle animation."""
        nonlocal particle_xy, particle_age, trail, base_wind_rgb_u8, last_anim_t
        rng = np.random.default_rng(1337)
        n = int(max(500, min(int(n), 60000)))
        particle_xy = np.empty((n, 2), dtype=np.float32)
        particle_xy[:, 0] = rng.uniform(0, w - 1, size=n).astype(np.float32)
        particle_xy[:, 1] = rng.uniform(0, h - 1, size=n).astype(np.float32)
        particle_age = rng.integers(0, 80, size=n, dtype=np.int32)
        trail = np.zeros((h, w), dtype=np.float32)
        base_wind_rgb_u8 = None
        last_anim_t = time.perf_counter()

    def _init_oc_particles(h: int, w: int, n: int, ocean_mask: np.ndarray) -> None:
        """Initialize ocean current particles, spawning only inside ocean cells."""
        nonlocal oc_particle_xy, oc_particle_age, oc_trail, oc_base_rgb_u8, oc_last_anim_t
        rng = np.random.default_rng(2024)
        n = int(max(200, min(int(n), 30000)))
        oc_particle_xy = np.empty((n, 2), dtype=np.float32)
        ocean_ys, ocean_xs = np.where(ocean_mask)
        if len(ocean_ys) > 0:
            idx = rng.integers(0, len(ocean_ys), size=n)
            oc_particle_xy[:, 0] = ocean_xs[idx].astype(np.float32)
            oc_particle_xy[:, 1] = ocean_ys[idx].astype(np.float32)
        else:
            oc_particle_xy[:, 0] = rng.uniform(0, w - 1, size=n).astype(np.float32)
            oc_particle_xy[:, 1] = rng.uniform(0, h - 1, size=n).astype(np.float32)
        oc_particle_age = rng.integers(0, 80, size=n, dtype=np.int32)
        oc_trail = np.zeros((h, w), dtype=np.float32)
        oc_base_rgb_u8 = None
        oc_last_anim_t = time.perf_counter()

    def _wind_uv_for_display() -> tuple[np.ndarray, np.ndarray] | None:
        """Get (u,v) from sim_state for wind particle animation."""
        nonlocal sim_state
        if sim_state is None or sim_state.wind_u is None or sim_state.wind_v is None:
            return None
        return sim_state.wind_u, sim_state.wind_v

    def _update_wind_particles() -> None:
        """Animate wind particles in map or globe mode using a fading trail buffer."""
        nonlocal tk_img, particle_xy, particle_age, trail, base_wind_rgb_u8, last_wind_key, last_anim_t, particle_anim_running, display_scale_x, display_scale_y, last_particle_mode
        _mode = mode_var.get()
        if view_var.get() != "Wind Particles" or _mode not in ("map", "globe"):
            particle_anim_running = False
            return
        # Only animate while the simulation is running (Start). Otherwise, show a static frame.
        if not sim_running or sim_paused:
            uv = _wind_uv_for_display()
            if uv is None:
                return
            u, v = uv
            speed = np.hypot(u, v).astype(np.float32)
            wind_rgb = wind_speed_to_rgb(speed)
            _stex = sim_state.elevation if (sim_state is not None and sim_state.elevation is not None) else None
            if _stex is not None:
                base_rgb = np.clip(0.60 * colorize(_stex) + 0.40 * wind_rgb, 0.0, 1.0)
            else:
                base_rgb = wind_rgb
            out = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            if _mode == "globe":
                out_f = out.astype(np.float32) / 255.0
                _, _ekey = get_elevation_cache()
                if _ekey is not None and isinstance(_ekey, tuple) and len(_ekey) >= 1 and _ekey[0] == "loaded":
                    out_f = np.fliplr(out_f)
                new_img = project_equirect_on_globe(out_f, size=size, radius=0.96, rot=(yaw, pitch, roll))
                canvas.config(width=size, height=size)
            else:
                new_img = Image.fromarray(out)
                canvas.config(width=out.shape[1], height=out.shape[0])
            _alloc_w = canvas.winfo_width()
            _alloc_h = canvas.winfo_height()
            _img_w, _img_h = new_img.size
            if _alloc_w > 1 and _alloc_h > 1:
                _scale = min(_alloc_w / _img_w, _alloc_h / _img_h)
                if abs(_scale - 1.0) > 0.005:
                    new_img = new_img.resize((max(1, int(_img_w * _scale)), max(1, int(_img_h * _scale))), Image.NEAREST)
                display_scale_x = new_img.width / _img_w
                display_scale_y = new_img.height / _img_h
            tk_img = ImageTk.PhotoImage(new_img)
            canvas.itemconfig(img_id, image=tk_img)
            particle_anim_running = False
            return

        # Use simulation elevation for dimensions (required for consistent indexing).
        if sim_state is None or sim_state.elevation is None:
            return
        tex = sim_state.elevation
        h, w = tex.shape
        _cur_mode = time_scale_var.get()
        if (trail is None or trail.shape != (h, w) or particle_xy is None or particle_age is None
                or last_particle_mode != _cur_mode):
            # Particle count: tie to "Arrows" control, scale up for particles, then
            # throttle down at coarser time-scale modes (decorative-only, see above).
            _init_particles(h, w, n=int(int(wind_arrows_var.get()) * 2 * _particle_count_scale()))
            last_particle_mode = _cur_mode

        uv = _wind_uv_for_display()
        if uv is None:
            return
        u, v = uv
        # Cache blended base (terrain + wind speed), recompute only if u/v identity changes
        uv_key = (id(u), id(v), h, w)
        if base_wind_rgb_u8 is None or last_wind_key != uv_key:
            speed = np.hypot(u, v).astype(np.float32)
            wind_rgb = wind_speed_to_rgb(speed)
            terrain_rgb = colorize(tex)
            base_rgb = np.clip(0.60 * terrain_rgb + 0.40 * wind_rgb, 0.0, 1.0)
            base_wind_rgb_u8 = (base_rgb * 255).astype(np.uint8)
            last_wind_key = uv_key

        # Time step (seconds) -> normalize to a stable "frames" unit
        now = time.perf_counter()
        dt_wall = now - last_anim_t
        if dt_wall < 0.05:
            root.after(50, _update_wind_particles)
            return
        dt = max(1e-3, min(0.08, dt_wall))
        last_anim_t = now

        # Fade trail
        trail *= 0.92

        # Advect particles (vectorized)
        xy = particle_xy
        x = xy[:, 0]
        y = xy[:, 1]
        xi = np.clip(x.astype(np.int32), 0, w - 1)
        yi = np.clip(y.astype(np.int32), 0, h - 1)
        uu = u[yi, xi].astype(np.float32)
        vv = v[yi, xi].astype(np.float32)
        sp = np.hypot(uu, vv) + 1e-6

        # Convert wind speed to pixels per tick (heuristic). Scale control acts like a multiplier.
        vmax = 25.0
        px_step = (float(wind_scale_var.get()) * 6.0) * (sp / vmax)
        dx = (uu / sp) * px_step
        dy = (-vv / sp) * px_step
        x1 = (x + dx).astype(np.float32)
        y1 = (y + dy).astype(np.float32)

        # Wrap in longitude, respawn if off-map in latitude
        x1 = np.mod(x1, float(w))
        alive = (y1 >= 0.0) & (y1 < float(h))

        # Intensity by speed; brighten fast flow
        inten = np.clip(sp / vmax, 0.0, 1.0) ** 0.6
        inten = inten.astype(np.float32) * 0.9

        # Deposit a short streak along motion (4 samples), additive then clamp
        samples = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
        xs = x[:, None] + (x1 - x)[:, None] * samples[None, :]
        ys = y[:, None] + (y1 - y)[:, None] * samples[None, :]
        xs = np.mod(xs, float(w))
        # only deposit for in-bounds y samples
        ys_clip = np.clip(ys, 0.0, float(h - 1))
        xi_s = xs.astype(np.int32).ravel()
        yi_s = ys_clip.astype(np.int32).ravel()
        w_s = np.repeat(inten, samples.size)
        # Add with clip (fast, avoids Python loops)
        trail[yi_s, xi_s] = np.clip(trail[yi_s, xi_s] + w_s * 0.4, 0.0, 1.0)

        # Update particle positions and ages
        particle_xy[:, 0] = x1
        particle_xy[:, 1] = np.where(alive, y1, y)  # keep y for dead until respawn
        particle_age -= 1

        # Respawn: dead age or out-of-bounds
        dead = (particle_age <= 0) | (~alive)
        if np.any(dead):
            rng = np.random.default_rng(int(now * 1000) & 0xFFFFFFFF)
            particle_xy[dead, 0] = rng.uniform(0, w - 1, size=int(np.sum(dead))).astype(np.float32)
            particle_xy[dead, 1] = rng.uniform(0, h - 1, size=int(np.sum(dead))).astype(np.float32)
            particle_age[dead] = rng.integers(40, 120, size=int(np.sum(dead)), dtype=np.int32)

        # Composite: base wind + white trails (brightness indicates speed)
        out = base_wind_rgb_u8.copy()
        streak = (np.clip(trail, 0.0, 1.0) * 255.0).astype(np.uint8)
        # Additive blend into RGB
        out[..., 0] = np.clip(out[..., 0].astype(np.int16) + streak.astype(np.int16), 0, 255).astype(np.uint8)
        out[..., 1] = np.clip(out[..., 1].astype(np.int16) + streak.astype(np.int16), 0, 255).astype(np.uint8)
        out[..., 2] = np.clip(out[..., 2].astype(np.int16) + streak.astype(np.int16), 0, 255).astype(np.uint8)

        if _mode == "globe":
            out_f = out.astype(np.float32) / 255.0
            _, _ekey = get_elevation_cache()
            if _ekey is not None and isinstance(_ekey, tuple) and len(_ekey) >= 1 and _ekey[0] == "loaded":
                out_f = np.fliplr(out_f)
            new_img = project_equirect_on_globe(out_f, size=size, radius=0.96, rot=(yaw, pitch, roll))
            canvas.config(width=size, height=size)
        else:
            new_img = Image.fromarray(out)
            canvas.config(width=w, height=h)
        _alloc_w = canvas.winfo_width()
        _alloc_h = canvas.winfo_height()
        _img_w, _img_h = new_img.size
        if _alloc_w > 1 and _alloc_h > 1:
            _scale = min(_alloc_w / _img_w, _alloc_h / _img_h)
            if abs(_scale - 1.0) > 0.005:
                new_img = new_img.resize((max(1, int(_img_w * _scale)), max(1, int(_img_h * _scale))), Image.NEAREST)
            display_scale_x = new_img.width / _img_w
            display_scale_y = new_img.height / _img_h
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)

        # Schedule next frame
        root.after(50, _update_wind_particles)

    def _update_ocean_particles() -> None:
        """Animate ocean current particles with a fading trail buffer (map mode only)."""
        nonlocal tk_img, oc_particle_xy, oc_particle_age, oc_trail, oc_base_rgb_u8, oc_last_uv_key, oc_last_anim_t, oc_anim_running, display_scale_x, display_scale_y, oc_last_particle_mode
        if view_var.get() != "Ocean Currents" or mode_var.get() != "map":
            oc_anim_running = False
            return

        if sim_state is None or sim_state.elevation is None:
            root.after(100, _update_ocean_particles)
            return
        tex = sim_state.elevation
        h, w = tex.shape
        ocean_mask, _ = get_masks(tex)

        # Fetch ocean current u, v (reuse shared cache)
        use_sim_data = sim_state is not None and sim_running
        day = int(sim_state.day_of_year) if sim_state is not None else 80
        tdays = float(sim_state.total_days) if sim_state is not None else float(day)
        wu = sim_state.wind_u if use_sim_data else None
        wv = sim_state.wind_v if use_sim_data else None
        ckey = (tex.shape, "oc_particles", int(tdays))
        if _OCEAN_CURRENT_CACHE["key"] != ckey:
            u, v = generate_ocean_currents(tex, wind_u=wu, wind_v=wv, day_of_year=day, time_days=tdays)
            _OCEAN_CURRENT_CACHE.update({"key": ckey, "u": u, "v": v})
        else:
            u, v = _OCEAN_CURRENT_CACHE["u"], _OCEAN_CURRENT_CACHE["v"]

        # Initialise particle buffers when needed (also reinit on time-scale mode change,
        # so particle count re-throttles without requiring a resolution change).
        _oc_cur_mode = time_scale_var.get()
        if (oc_trail is None or oc_trail.shape != (h, w) or oc_particle_xy is None
                or oc_last_particle_mode != _oc_cur_mode):
            _init_oc_particles(
                h, w, n=int(int(wind_arrows_var.get()) * 2 * _particle_count_scale()), ocean_mask=ocean_mask
            )
            oc_last_particle_mode = _oc_cur_mode

        # Rebuild base image when velocity field changes
        uv_key = (id(u), id(v), h, w)
        if oc_base_rgb_u8 is None or oc_last_uv_key != uv_key:
            speed = np.hypot(u, v).astype(np.float32)
            terrain_rgb = colorize(tex)
            current_rgb = wind_speed_to_rgb(speed)
            mask3 = ocean_mask[..., None].astype(np.float32)
            # Land: pure terrain; ocean: 60% terrain + 40% current-speed colour
            base_rgb = np.clip(terrain_rgb * (1.0 - mask3 * 0.40) + current_rgb * mask3 * 0.40, 0.0, 1.0)
            oc_base_rgb_u8 = (base_rgb * 255).astype(np.uint8)
            oc_last_uv_key = uv_key

        # Throttle to 20 fps
        now = time.perf_counter()
        dt_wall = now - oc_last_anim_t
        if dt_wall < 0.05:
            root.after(50, _update_ocean_particles)
            return
        oc_last_anim_t = now

        # Fade trail
        oc_trail *= 0.93

        # Advect particles
        xy = oc_particle_xy
        x = xy[:, 0]
        y = xy[:, 1]
        xi = np.clip(x.astype(np.int32), 0, w - 1)
        yi = np.clip(y.astype(np.int32), 0, h - 1)
        uu = u[yi, xi].astype(np.float32)
        vv = v[yi, xi].astype(np.float32)
        sp = np.hypot(uu, vv) + 1e-6
        vmax = 2.0  # m/s — typical max surface current
        px_step = (float(wind_scale_var.get()) * 5.0) * (sp / vmax)
        dx = (uu / sp) * px_step
        dy = (-vv / sp) * px_step
        x1 = np.mod((x + dx).astype(np.float32), float(w))
        y1 = (y + dy).astype(np.float32)
        xi1 = np.clip(x1.astype(np.int32), 0, w - 1)
        yi1 = np.clip(y1.astype(np.int32), 0, h - 1)
        alive = (y1 >= 0.0) & (y1 < float(h)) & ocean_mask[yi1, xi1]

        # Deposit streak
        inten = (np.clip(sp / vmax, 0.0, 1.0) ** 0.6 * 0.9).astype(np.float32)
        samples = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
        xs = x[:, None] + (x1 - x)[:, None] * samples[None, :]
        ys = y[:, None] + (y1 - y)[:, None] * samples[None, :]
        xs = np.mod(xs, float(w))
        ys_clip = np.clip(ys, 0.0, float(h - 1))
        xi_s = xs.astype(np.int32).ravel()
        yi_s = ys_clip.astype(np.int32).ravel()
        w_s = np.repeat(inten, samples.size)
        oc_trail[yi_s, xi_s] = np.clip(oc_trail[yi_s, xi_s] + w_s * 0.4, 0.0, 1.0)

        # Update positions and ages
        oc_particle_xy[:, 0] = x1
        oc_particle_xy[:, 1] = np.where(alive, y1, y)
        oc_particle_age -= 1

        # Respawn dead/stranded particles back into ocean cells
        dead = (oc_particle_age <= 0) | (~alive)
        if np.any(dead):
            n_dead = int(np.sum(dead))
            ocean_ys, ocean_xs = np.where(ocean_mask)
            if len(ocean_ys) > 0:
                rng = np.random.default_rng(int(now * 1000) & 0xFFFFFFFF)
                idx = rng.integers(0, len(ocean_ys), size=n_dead)
                oc_particle_xy[dead, 0] = ocean_xs[idx].astype(np.float32)
                oc_particle_xy[dead, 1] = ocean_ys[idx].astype(np.float32)
                oc_particle_age[dead] = rng.integers(40, 120, size=n_dead, dtype=np.int32)

        # Composite trails onto base image
        out = oc_base_rgb_u8.copy()
        streak = (np.clip(oc_trail, 0.0, 1.0) * 255.0).astype(np.uint8)
        out[..., 0] = np.clip(out[..., 0].astype(np.int16) + streak.astype(np.int16), 0, 255).astype(np.uint8)
        out[..., 1] = np.clip(out[..., 1].astype(np.int16) + streak.astype(np.int16), 0, 255).astype(np.uint8)
        out[..., 2] = np.clip(out[..., 2].astype(np.int16) + streak.astype(np.int16), 0, 255).astype(np.uint8)

        new_img = Image.fromarray(out)
        canvas.config(width=w, height=h)
        _alloc_w = canvas.winfo_width()
        _alloc_h = canvas.winfo_height()
        _img_w, _img_h = new_img.size
        if _alloc_w > 1 and _alloc_h > 1:
            _scale = min(_alloc_w / _img_w, _alloc_h / _img_h)
            if abs(_scale - 1.0) > 0.005:
                new_img = new_img.resize((max(1, int(_img_w * _scale)), max(1, int(_img_h * _scale))), Image.NEAREST)
            display_scale_x = new_img.width / _img_w
            display_scale_y = new_img.height / _img_h
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)

        root.after(50, _update_ocean_particles)

    def export_data():
        """Export simulation time series data."""
        if not diagnostics.history:
            messagebox.showinfo("Info", "No simulation data to export. Please run the simulation first.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Simulation Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                if filename.endswith('.json'):
                    filepath = diagnostics.export_time_series(filename, format='json')
                else:
                    filepath = diagnostics.export_time_series(filename, format='csv')
                messagebox.showinfo("Export Complete", f"Data exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
    
    def run_benchmark():
        # Without the nonlocal, `sim_running = False` bound a LOCAL variable and
        # the simulation thread kept stepping while the benchmark ran on the
        # same state (race + double-stepping).
        nonlocal sim_running
        if sim_state is None:
            messagebox.showinfo("Info", "Please start or initialize simulation first.")
            return
        
        sim_running = False
        sim_status_var.set("Benchmarking...")
        root.update()
        
        LOG.info("Starting 1-year benchmark...")

        # --- T_BASE PROFILE (instant, no simulation needed) ---
        # This shows whether the temperature TARGETS are calibrated correctly
        # before we even run a single step.  If T_base is wrong, no amount of
        # tuning other parameters will fix the climate.
        from diagnostics import compute_t_base_profile, print_t_base_report
        print_t_base_report(compute_t_base_profile())

        # --- RUN 365 days in monthly chunks to collect ice snapshots ---
        # Month day-counts and cumulative start days (non-leap year)
        MONTHLY_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        _MONTH_START = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        # Determine which calendar month the simulation is currently in so the
        # monthly labels align with the simulation's actual day_of_year.
        _doy = sim_state.day_of_year % 365.0
        _start_month_idx = max(
            (i for i, s in enumerate(_MONTH_START) if s <= _doy),
            default=0,
        )
        monthly_ice_nh: list[float] = []
        monthly_ice_sh: list[float] = []
        monthly_T55N: list[float] = []  # key diagnostic latitude
        monthly_T65N: list[float] = []
        monthly_T75N: list[float] = []

        current = sim_state
        for month_i, mdays in enumerate(MONTHLY_DAYS):
            states, _ = simulate_multiple_steps(current, total_days=float(mdays), step_days=1.0)
            current = states[-1]
            # Monthly ice snapshot
            snap = diagnostics.analyze_snapshot(current)
            monthly_ice_nh.append(snap.get("ice_frac_nh", 0.0))
            monthly_ice_sh.append(snap.get("ice_frac_sh", 0.0))
            # Zonal mean temperature at key NH latitudes (use air temperature)
            T = current.air_temperature if current.air_temperature is not None else current.temperature
            if T is not None:
                H = T.shape[0]
                lat_rows = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
                def _zonal_T(lat_c: float) -> float:
                    idx = int(np.argmin(np.abs(lat_rows - lat_c)))
                    return float(np.mean(T[idx, :])) - 273.15
                monthly_T55N.append(_zonal_T(55.0))
                monthly_T65N.append(_zonal_T(65.0))
                monthly_T75N.append(_zonal_T(75.0))

        final_state = current

        # --- PRINT MONTHLY ICE / TEMPERATURE EVOLUTION ---
        _ALL_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        # Rotate month names so index 0 = the calendar month the benchmark started in
        MONTH_NAMES = [_ALL_MONTHS[(_start_month_idx + i) % 12] for i in range(12)]
        print(f"\n--- MONTHLY ICE & HIGH-LAT TEMPERATURE EVOLUTION (starting ~{_ALL_MONTHS[_start_month_idx]}, day {_doy:.0f}) ---")
        print(f"{'Month':>5} | {'NH ice%':>7} | {'SH ice%':>7} | {'T55N(°C)':>9} | {'T65N(°C)':>9} | {'T75N(°C)':>9}")
        print("-" * 62)
        for i, name in enumerate(MONTH_NAMES):
            ni = monthly_ice_nh[i]*100 if i < len(monthly_ice_nh) else float('nan')
            si = monthly_ice_sh[i]*100 if i < len(monthly_ice_sh) else float('nan')
            t55 = monthly_T55N[i] if i < len(monthly_T55N) else float('nan')
            t65 = monthly_T65N[i] if i < len(monthly_T65N) else float('nan')
            t75 = monthly_T75N[i] if i < len(monthly_T75N) else float('nan')
            print(f"{name:>5} | {ni:>7.1f} | {si:>7.1f} | {t55:>+9.1f} | {t65:>+9.1f} | {t75:>+9.1f}")
        print("-" * 62)
        nh_min = min(monthly_ice_nh)*100 if monthly_ice_nh else 0
        nh_max = max(monthly_ice_nh)*100 if monthly_ice_nh else 0
        print(f"  NH ice range: {nh_min:.1f}% – {nh_max:.1f}%  (Earth: ~3% summer to ~8% winter)")
        t55_range = f"{min(monthly_T55N):+.1f}°C – {max(monthly_T55N):+.1f}°C" if monthly_T55N else "N/A"
        t65_range = f"{min(monthly_T65N):+.1f}°C – {max(monthly_T65N):+.1f}°C" if monthly_T65N else "N/A"
        print(f"  T55N seasonal range: {t55_range}  (Earth: ~-5°C winter to +15°C summer)")
        print(f"  T65N seasonal range: {t65_range}  (Earth: ~-15°C winter to +10°C summer)")
        print("-----------------------------------------------------\n")

        # Analyze final state snapshot.
        stats = diagnostics.analyze_snapshot(final_state)
        diagnostics.print_report(stats)
        # Wind speed and direction magnitudes vs Earth
        diagnostics.print_wind_report(stats)
        # Circulation validation (surface 3-cell proxies)
        circ = diagnostics.analyze_circulation(stats)
        diagnostics.print_circulation_report(circ)
        # Latitude band temperature/precip comparison
        from diagnostics import compute_latitude_band_stats, print_latitude_band_report
        band_stats = compute_latitude_band_stats(final_state)
        print_latitude_band_report(band_stats)
        # Sea ice extent vs Earth references
        diagnostics.print_ice_report(stats)

        cs = stats.get("circulation_score", 0.0)
        messagebox.showinfo("Benchmark Complete",
            f"Global Mean Temp: {stats['global_mean_temp']:.1f} K\n"
            f"Equator-Pole Gradient (N): {stats['gradient_north']:.1f} K  (Earth 45-60 K)\n"
            f"Circulation Score: {cs:.2f}  (higher = more Earth-like)\n"
            "Check console for full report.")
        
        sim_status_var.set("Stopped")

    # Export Data / Benchmark are exposed via the Simulation menu (see menu
    # setup near the end of main()) rather than as always-visible buttons.

    # --- State save/load controls (File menu is the primary entry point; see
    # menu setup near the end of main() for Open/Save/Save As/Exit) ---
    auto_save_var = tk.BooleanVar(value=_auto_save_enabled)
    save_info_var = tk.StringVar(value="")

    def _refresh_save_info() -> None:
        if _current_state_path.exists():
            size_kb = _current_state_path.stat().st_size / 1024
            save_info_var.set(f"{_current_state_path.name} ({size_kb:.0f} KB)")
        else:
            save_info_var.set(f"{_current_state_path.name} (not saved yet)")

    def _do_save_state() -> None:
        """Save to the currently active file (File>Open / File>Save As target)."""
        nonlocal sim_state
        if sim_state is None:
            messagebox.showinfo("Save State", "No simulation state to save. Start the simulation first.")
            return
        try:
            save_state(sim_state, _current_state_path)
            _refresh_save_info()
            total_years = sim_state.total_days / 365.2422
            messagebox.showinfo("Save State", f"State saved to {_current_state_path.name}.\nSimulation day {sim_state.total_days:.0f} ({total_years:.2f} years)")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _do_save_state_as() -> None:
        """Prompt for a new file path and save the current state there."""
        nonlocal sim_state, _current_state_path
        if sim_state is None:
            messagebox.showinfo("Save State As", "No simulation state to save. Start the simulation first.")
            return
        saves_dir = Path("saves")
        saves_dir.mkdir(parents=True, exist_ok=True)
        path_str = filedialog.asksaveasfilename(
            title="Save State As",
            initialdir=str(saves_dir),
            initialfile=_current_state_path.name,
            defaultextension=".pkl",
            filetypes=[("Planet State", "*.pkl"), ("All files", "*.*")],
        )
        if not path_str:
            return
        _current_state_path = Path(path_str)
        _do_save_state()

    def _do_load_state() -> None:
        """Prompt for a state file to open and load it."""
        nonlocal sim_state, sim_running, sim_paused, sim_thread, _current_state_path
        saves_dir = Path("saves")
        saves_dir.mkdir(parents=True, exist_ok=True)
        path_str = filedialog.askopenfilename(
            title="Open State",
            initialdir=str(saves_dir),
            filetypes=[("Planet State", "*.pkl"), ("All files", "*.*")],
        )
        if not path_str:
            return
        chosen_path = Path(path_str)
        # Stop and discard the thread so Start recreates it from the loaded state
        if sim_thread and sim_thread.is_alive():
            sim_thread.stop()
            sim_thread = None
        sim_running = False
        sim_paused = False
        try:
            sim_state = load_state(chosen_path)
            _current_state_path = chosen_path
            total_years = sim_state.total_days / 365.2422
            sim_status_var.set("Stopped")
            year = int(sim_state.total_days // 365.2422) + 1
            month = int((sim_state.day_of_year / 365.2422) * 12) + 1
            sim_cycle_var.set(f"Y{year} M{month}")
            _refresh_save_info()
            root.title(f"Sphere {size}x{size} - {chosen_path.name}")
            render()
            messagebox.showinfo("Open State", f"State loaded from {chosen_path.name}.\nSimulation day {sim_state.total_days:.0f} ({total_years:.2f} years)")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _on_auto_save_toggle() -> None:
        nonlocal _auto_save_enabled
        _auto_save_enabled = auto_save_var.get()

    ttk.Checkbutton(sim_tab, text="Auto-save/load current file", variable=auto_save_var, command=_on_auto_save_toggle).pack(fill="x", pady=2)
    ttk.Label(status_bar, text="File:", foreground="gray").pack(side="left", padx=(8, 2))
    ttk.Label(status_bar, textvariable=save_info_var, foreground="gray").pack(side="left", padx=2)
    _refresh_save_info()

    def _init_sim_state_from_elevation(elev: np.ndarray) -> None:
        """Initialize sim_state immediately (but do not start stepping)."""
        nonlocal sim_state, sim_running, sim_paused
        sim_state = create_initial_state(
            elev,
            day_of_year=80.0,
            wind_block_size=int(wind_block_size_var.get()),
        )
        sim_running = False
        sim_paused = False
        sim_status_var.set("Stopped")
        diagnostics.history.clear()
        diagnostics.component_history.clear()
        diagnostics.total_days = 0.0
        sim_cycle_var.set("Y1 M1")
        graphs_controller.reset()
    
    # Simulation controls (Simulation tab)
    sim_status_var = tk.StringVar(value="Stopped")
    sim_cycle_var = tk.StringVar(value="Year: 1")
    sim_buttons_row = ttk.Frame(sim_tab)
    sim_buttons_row.pack(fill="x", pady=2)
    ttk.Button(sim_buttons_row, text="Start", command=lambda: start_simulation()).grid(row=0, column=0, sticky="ew", padx=1, pady=1)
    ttk.Button(sim_buttons_row, text="Stop", command=lambda: stop_simulation()).grid(row=0, column=1, sticky="ew", padx=1, pady=1)
    ttk.Button(sim_buttons_row, text="Pause", command=lambda: pause_simulation()).grid(row=1, column=0, sticky="ew", padx=1, pady=1)
    ttk.Button(sim_buttons_row, text="Reset", command=lambda: reset_simulation()).grid(row=1, column=1, sticky="ew", padx=1, pady=1)
    sim_buttons_row.columnconfigure(0, weight=1)
    sim_buttons_row.columnconfigure(1, weight=1)

    def on_graphs_toggle():
        graphs_controller.set_enabled(graphs_enabled_var.get())
    ttk.Checkbutton(sim_tab, text="Graphs", variable=graphs_enabled_var, command=on_graphs_toggle).pack(fill="x", pady=2)

    # Time scale dropdown — each option maps to a TimeScaleMode
    time_scale_options = {
        "1 Day":   TimeScaleMode.DAILY,
        "1 Week":  TimeScaleMode.WEEKLY,
        "1 Month": TimeScaleMode.MONTHLY,
        "1 Year":  TimeScaleMode.ANNUAL,
    }
    time_scale_var = tk.StringVar(value="1 Day")
    def on_time_scale_change(*_args):
        nonlocal sim_speed
        mode = time_scale_options[time_scale_var.get()]
        # sim_speed is kept as a rough "days per frame" for any legacy display code
        sim_speed = {"1 Day": 1, "1 Week": 7, "1 Month": 30, "1 Year": 365}.get(
            time_scale_var.get(), 1
        )
        if sim_thread is not None and sim_thread.is_alive():
            sim_thread.update_time_scale(mode)
    time_scale_var.trace_add("write", on_time_scale_change)
    speed_row = ttk.Frame(sim_tab)
    speed_row.pack(fill="x", pady=2)
    ttk.Label(speed_row, text="Speed:").pack(side="left")
    ttk.Combobox(
        speed_row, textvariable=time_scale_var, state="readonly", width=8,
        values=list(time_scale_options.keys()),
    ).pack(side="left", padx=(4, 0))

    # Status bar: sim status/cycle + year-progress bar (always visible regardless
    # of which sidebar tab is open)
    _MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    cycle_progress_var = tk.DoubleVar(value=0.0)
    cycle_detail_var = tk.StringVar(value="")
    ttk.Label(status_bar, text="Sim:").pack(side="left", padx=(0, 2))
    sim_status_label = ttk.Label(status_bar, textvariable=sim_status_var, width=12, anchor="w")
    sim_status_label.pack(side="left")
    sim_cycle_label = ttk.Label(status_bar, textvariable=sim_cycle_var, width=10, anchor="w")
    sim_cycle_label.pack(side="left", padx=(4, 8))
    cycle_progress_bar = ttk.Progressbar(
        status_bar, variable=cycle_progress_var,
        maximum=100.0, mode="determinate", length=120,
    )
    cycle_progress_bar.pack(side="left", padx=(0, 8))
    ttk.Label(status_bar, textvariable=cycle_detail_var, width=26, anchor="w",
              font=("Courier", 8)).pack(side="left")

    # Wind controls (View tab — they affect wind/particle rendering)
    wind_arrows_var = tk.IntVar(value=int(settings.get("wind_arrows", default_settings["wind_arrows"])))
    wind_scale_var = tk.DoubleVar(value=float(settings.get("wind_scale", default_settings["wind_scale"])))
    wind_block_size_var = tk.IntVar(value=int(settings.get("wind_block_size", default_settings["wind_block_size"])))
    # Precipitation simulation removed.
    def add_wind_controls(parent):
        frm = ttk.Frame(parent)
        frm.pack(fill="x", pady=2)
        def add(sub_parent, label, var, width=6):
            f = ttk.Frame(sub_parent); f.pack(side="left", padx=4)
            ttk.Label(f, text=label).pack(side="left")
            ttk.Entry(f, textvariable=var, width=width).pack(side="left")
        add(frm, "Arrows", wind_arrows_var)
        add(frm, "Scale", wind_scale_var)
        add(frm, "WindBS", wind_block_size_var, width=4)
        return frm
    wind_controls = add_wind_controls(sim_tab)

    # Terrain parameter inputs (Terrain tab)
    seed_var = tk.IntVar(value=int(settings["seed"]))
    octaves_var = tk.IntVar(value=int(settings["octaves"]))
    freq_var = tk.DoubleVar(value=float(settings["freq"]))
    lac_var = tk.DoubleVar(value=float(settings["lac"]))
    gain_var = tk.DoubleVar(value=float(settings["gain"]))

    def add_labeled_entry(parent, label, var, width=6):
        frm = ttk.Frame(parent)
        frm.pack(fill="x", pady=1)
        ttk.Label(frm, text=label, width=6).pack(side="left")
        entry = ttk.Entry(frm, textvariable=var, width=width)
        entry.pack(side="left")
        return entry

    seed_entry = add_labeled_entry(terrain_tab, "Seed", seed_var)
    octaves_entry = add_labeled_entry(terrain_tab, "Oct", octaves_var)
    freq_entry = add_labeled_entry(terrain_tab, "Freq", freq_var)
    lac_entry = add_labeled_entry(terrain_tab, "Lac", lac_var)
    gain_entry = add_labeled_entry(terrain_tab, "Gain", gain_var)

    def load_heightmap():
        """Load a heightmap from a .TIF file."""
        nonlocal terrain_mode, loaded_heightmap_path, sim_state, sim_running, sim_paused, tk_img
        
        filepath = filedialog.askopenfilename(
            title="Open Heightmap",
            filetypes=[("TIFF Images", "*.tif *.tiff"), ("All Files", "*.*")]
        )
        if not filepath:
            return
        
        try:
            with log_time(f"Loading heightmap from {filepath}"):
                img = Image.open(filepath)
                # Handle both 8-bit (L) and 16-bit (I;16) images
                if img.mode == 'I;16':
                    arr = np.array(img, dtype=np.float32)
                elif img.mode == 'I':
                    arr = np.array(img, dtype=np.float32)
                else:
                    img = img.convert('L')
                    arr = np.array(img, dtype=np.float32)
                
                # Validate dimensions: must be 512 height x 1024 width
                if arr.shape != (size, size * 2):
                    messagebox.showerror(
                        "Invalid Dimensions",
                        f"Heightmap must be {size * 2}x{size} pixels (width x height).\n"
                        f"Got: {arr.shape[1]}x{arr.shape[0]} pixels."
                    )
                    return
                
                # Ocean threshold for 16-bit DEM: 4592-8546 = ocean
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))
                LOG.info(f"Raw heightmap range: min={min_val}, max={max_val}")
                
                # Ocean threshold: lower value = less ocean, more low-elevation land
                ocean_threshold = 8070.0  # Lowered to classify more shallow areas as land
                LOG.info(f"Ocean threshold: {ocean_threshold} (pixels <= {ocean_threshold} = ocean)")
                
                # Remap: <=threshold -> 0.0, >threshold -> scale to [0, 1]
                arr = np.maximum(0.0, (arr - ocean_threshold) / (max_val - ocean_threshold))
                
                # Set the elevation cache
                set_elevation_cache(arr, key=("loaded", filepath))
                invalidate_view_caches()
                
                # Update state
                terrain_mode = "loaded"
                loaded_heightmap_path = filepath
                
                # Disable noise parameter controls
                seed_entry.config(state="disabled")
                octaves_entry.config(state="disabled")
                freq_entry.config(state="disabled")
                lac_entry.config(state="disabled")
                gain_entry.config(state="disabled")
                
                # Reset simulation (initialize immediately, but not running)
                _init_sim_state_from_elevation(arr)
                
                # Update title
                root.title(f"Sphere {size}x{size} - Loaded Heightmap")
                
                LOG.info(f"Loaded heightmap: {filepath}")
                render()
                
        except Exception as e:
            messagebox.showerror("Error Loading Heightmap", f"Failed to load heightmap:\n{str(e)}")
            LOG.error(f"Failed to load heightmap: {e}")
    
    def use_procedural_terrain():
        """Switch back to procedural terrain generation."""
        nonlocal terrain_mode, loaded_heightmap_path, sim_state, sim_running, sim_paused, tk_img
        
        if terrain_mode == "procedural":
            return
        
        # Clear loaded terrain
        clear_elevation_cache()
        invalidate_view_caches()
        terrain_mode = "procedural"
        loaded_heightmap_path = None
        
        # Enable noise parameter controls
        seed_entry.config(state="normal")
        octaves_entry.config(state="normal")
        freq_entry.config(state="normal")
        lac_entry.config(state="normal")
        gain_entry.config(state="normal")
        
        # Reset simulation (initialize immediately, but not running)
        tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
        _init_sim_state_from_elevation(tex)
        
        # Update title
        root.title(f"Sphere {size}x{size} (262,144 cells)")
        
        LOG.info("Switched to procedural terrain generation")
        render()
    
    # Menu bar is assembled near the end of main(), once load_heightmap,
    # use_procedural_terrain, do_regen, export_data, run_benchmark, and the
    # state save/load + on_close callbacks all exist.

    def do_regen():
        nonlocal tk_img, terrain_mode, display_scale_x, display_scale_y, oc_anim_running, _last_render_arr
        # Only clear cache if using procedural terrain
        if terrain_mode == "procedural":
            clear_elevation_cache()
        _PRECIP_VIEW_CACHE.update({"key": None, "P": None})
        invalidate_view_caches()
        p = {
            "seed": int(seed_var.get()),
            "octaves": int(octaves_var.get()),
            "freq": float(freq_var.get()),
            "lac": float(lac_var.get()),
            "gain": float(gain_var.get()),
        }
        if mode_var.get() == "globe":
            day = int(sim_state.day_of_year) if (sim_state is not None and sim_running) else 80
            view_name = view_var.get()
            if view_name == "Wind Arrows" or view_name == "Wind Particles":
                view_name = "Wind"
            new_img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll), view=view_name, day_of_year=day, **p)
            canvas.config(width=size, height=size)
        else:
            tex = ensure_elevation(size, **p)
            if view_var.get() == "Wind Particles":
                # Start animation loop; render() will keep it going.
                nonlocal particle_anim_running
                if not particle_anim_running:
                    particle_anim_running = True
                    _update_wind_particles()
                return
            if view_var.get() == "Wind Arrows":
                # Use simulated wind if available, otherwise generate synthetic wind
                if sim_state is not None and sim_running and sim_state.wind_u is not None:
                    u, v = sim_state.wind_u, sim_state.wind_v
                else:
                    wkey = (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()), "bilinear")
                    if _WIND_CACHE["key"] != wkey:
                        with log_time("Generate wind field+arrows"):
                            u, v = generate_wind_field(*tex.shape, elevation=tex, upsample="bilinear", weather_amp=0.35, debug_log=False)
                            _WIND_CACHE.update({"key": wkey, "u": u, "v": v})
                    else:
                        u, v = _WIND_CACHE["u"], _WIND_CACHE["v"]
                speed = np.hypot(u, v).astype(np.float32)
                base_rgb = np.clip(0.60 * colorize(tex) + 0.40 * wind_speed_to_rgb(speed), 0.0, 1.0)
                arrows = render_wind_arrows(*tex.shape, u, v, target_arrows=int(wind_arrows_var.get()), scale=float(wind_scale_var.get()))
                comb = np.clip(base_rgb + arrows, 0.0, 1.0)
                arr = (comb * 255).astype(np.uint8)
            elif view_var.get() == "Ocean Currents":
                if not oc_anim_running:
                    oc_anim_running = True
                    _update_ocean_particles()
                return
            else:
                rgbf = colorize(tex)
                arr = (np.clip(rgbf, 0.0, 1.0) * 255).astype(np.uint8)
            _last_render_arr = arr
            new_img = Image.fromarray(arr)
            h, w = tex.shape
            canvas.config(width=w, height=h)
        _alloc_w = canvas.winfo_width()
        _alloc_h = canvas.winfo_height()
        _img_w, _img_h = new_img.size
        if _alloc_w > 1 and _alloc_h > 1:
            _scale = min(_alloc_w / _img_w, _alloc_h / _img_h)
            if abs(_scale - 1.0) > 0.005:
                new_img = new_img.resize((max(1, int(_img_w * _scale)), max(1, int(_img_h * _scale))), Image.NEAREST)
            display_scale_x = new_img.width / _img_w
            display_scale_y = new_img.height / _img_h
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)

    ttk.Button(terrain_tab, text="Regenerate", command=do_regen).pack(fill="x", pady=(6, 2))
    latlon_label = ttk.Label(status_bar, textvariable=latlon_var, width=60, anchor="e")
    latlon_label.pack(side="right")

    # Profile initial generation once at startup
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Try to load default heightmap, fall back to procedural if not found
    default_heightmap_path = r"D:\dev\planetsim\images\16_bit_dem_small_512.tif" #desktop location
    # default_heightmap_path = r"C:\dev\planetsim\images\16_bit_dem_small_512.tif" #laptop location
    try:
        with log_time(f"Loading default heightmap from {default_heightmap_path}"):
            heightmap_img = Image.open(default_heightmap_path)
            # Handle both 8-bit (L) and 16-bit (I;16) images
            if heightmap_img.mode == 'I;16':
                # 16-bit image: read directly as uint16
                arr = np.array(heightmap_img, dtype=np.float32)
            elif heightmap_img.mode == 'I':
                # 32-bit integer mode
                arr = np.array(heightmap_img, dtype=np.float32)
            else:
                # 8-bit or other: convert to grayscale
                heightmap_img = heightmap_img.convert('L')
                arr = np.array(heightmap_img, dtype=np.float32)
            
            # Validate dimensions
            if arr.shape == (size, size * 2):
                # Analyze 16-bit DEM
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))
                LOG.info(f"Raw heightmap range: min={min_val}, max={max_val}")
                
                # Ocean threshold: lower value = less ocean, more low-elevation land
                # Histogram shows: 7557-8546 has 363,132 pixels (bulk ocean)
                ocean_threshold = 8070.0  # Lowered to classify more shallow areas as land
                LOG.info(f"Ocean threshold: {ocean_threshold} (pixels <= {ocean_threshold} = ocean)")
                
                # Remap: <=8546 -> 0.0, >8546 -> scale to [0, 1]
                arr = np.maximum(0.0, (arr - ocean_threshold) / (max_val - ocean_threshold))
                LOG.info(f"Normalized range: min={float(np.min(arr)):.4f}, max={float(np.max(arr)):.4f}, mean={float(np.mean(arr)):.4f}")
                LOG.info(f"Pixels at 0.0: {np.sum(arr == 0.0)}, <0.02: {np.sum(arr < 0.02)}, 0.02-0.04: {np.sum((arr >= 0.02) & (arr < 0.04))}, >0.04: {np.sum(arr >= 0.04)}")
                
                # Set elevation cache
                set_elevation_cache(arr, key=("loaded", default_heightmap_path))
                invalidate_view_caches()
                
                # Update state
                terrain_mode = "loaded"
                loaded_heightmap_path = default_heightmap_path
                
                # Disable noise parameter controls
                seed_entry.config(state="disabled")
                octaves_entry.config(state="disabled")
                freq_entry.config(state="disabled")
                lac_entry.config(state="disabled")
                gain_entry.config(state="disabled")
                
                # Update title
                root.title(f"Sphere {size}x{size} - Loaded Heightmap")
                
                tex0 = arr
                LOG.info(f"Loaded default heightmap successfully")
            else:
                LOG.warning(f"Default heightmap has wrong dimensions {arr.shape}, using procedural")
                tex0 = ensure_elevation(size, seed=settings["seed"], octaves=settings["octaves"], freq=settings["freq"], lac=settings["lac"], gain=settings["gain"])
    except Exception as e:
        LOG.warning(f"Could not load default heightmap: {e}, using procedural")
        tex0 = ensure_elevation(size, seed=settings["seed"], octaves=settings["octaves"], freq=settings["freq"], lac=settings["lac"], gain=settings["gain"])

    # Initialize sim_state immediately (stopped) so wind particles can use sim winds even before Start.
    _init_sim_state_from_elevation(tex0)
    
    rgbf0 = colorize(tex0)
    arr0 = (np.clip(rgbf0, 0.0, 1.0) * 255).astype(np.uint8)
    _last_render_arr = arr0  # seed zoom preview before first render()
    img = Image.fromarray(arr0)
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    LOG.info("Startup profile (top 20):")
    stats.print_stats(20)
    tk_img = ImageTk.PhotoImage(img)
    # Map area: canvas on the left, legend panel on the right (unused window space).
    # Lives in the right pane of `body` so users can drag the sash to resize
    # the sidebar vs. the map.
    map_row = tk.Frame(body)
    body.add(map_row, weight=1)
    canvas = tk.Canvas(map_row, width=tex0.shape[1], height=tex0.shape[0], highlightthickness=0, bg="black")
    canvas.pack(side="left", fill="both", expand=True)
    img_id = canvas.create_image(0, 0, image=tk_img, anchor="nw")

    # --- Cursor tooltip: floating panel with zoom preview + text ---
    _ZOOM_CELLS = 21    # cells visible each axis; mouse wheel changes this
    _ZOOM_FACTOR = 8    # display pixels per simulation cell (fixed)
    _tooltip_var = tk.StringVar(value="")
    _zoom_tk_img = None  # keep reference to prevent GC
    _tooltip_win = tk.Toplevel(root)
    _tooltip_win.overrideredirect(True)
    _tooltip_win.withdraw()
    _tooltip_win.wm_attributes("-topmost", True)
    _tt_frame = tk.Frame(_tooltip_win, bg="#0d1117", bd=1, relief="solid")
    _tt_frame.pack()
    _zoom_canvas = tk.Canvas(_tt_frame, width=_ZOOM_CELLS * _ZOOM_FACTOR, height=_ZOOM_CELLS * _ZOOM_FACTOR,
                              bg="#000000", highlightthickness=0)
    _zoom_canvas.pack(padx=4, pady=(4, 2))
    _zoom_img_id = _zoom_canvas.create_image(0, 0, anchor="nw")
    tk.Frame(_tt_frame, bg="#2a2a3a", height=1).pack(fill="x", padx=4)
    tk.Label(
        _tt_frame,
        textvariable=_tooltip_var,
        bg="#0d1117",
        fg="#c9d1d9",
        font=("Courier", 9),
        justify="left",
        padx=8,
        pady=5,
    ).pack()

    # --- Biome legend panel ---
    # Shown only when the Biomes map view is active; hidden otherwise.
    from climate_averages import KOPPEN_NAMES as _KNAMES, KOPPEN_COLORS as _KCLR
    _LEG_BG  = "#1e1e2e"
    _LEG_FG  = "#d8d8f0"
    _LEG_DIM = "#77779a"
    legend_outer = tk.Frame(map_row, bg=_LEG_BG, padx=2, pady=2)
    # Not packed here — _on_view_change shows/hides it based on current view.
    tk.Label(legend_outer, text="Köppen Climate Zones",
             bg=_LEG_BG, fg="white", font=("TkDefaultFont", 9, "bold"),
             anchor="w").pack(fill="x", padx=6, pady=(6, 2))
    tk.Frame(legend_outer, bg="#4a4a6a", height=1).pack(fill="x", padx=6, pady=(0, 4))
    for _ci in range(1, 20):
        _rgb_f = _KCLR[_ci]
        _hex = "#{:02x}{:02x}{:02x}".format(
            int(_rgb_f[0] * 255), int(_rgb_f[1] * 255), int(_rgb_f[2] * 255)
        )
        _full = _KNAMES.get(_ci, f"Type {_ci}")
        _parts = _full.split(" - ", 1)
        _code_str = _parts[0]
        _desc = _parts[1] if len(_parts) > 1 else ""
        _row = tk.Frame(legend_outer, bg=_LEG_BG)
        _row.pack(fill="x", padx=6, pady=1)
        tk.Canvas(_row, width=13, height=13, bg=_hex, highlightthickness=0).pack(
            side="left", padx=(0, 5))
        tk.Label(_row, text=f"{_code_str:<5}{_desc}", bg=_LEG_BG, fg=_LEG_FG,
                 font=("Courier", 8), anchor="w").pack(side="left")
    tk.Frame(legend_outer, bg="#4a4a6a", height=1).pack(fill="x", padx=6, pady=(4, 2))
    tk.Label(legend_outer, text="Reclassified every 30 sim-days",
             bg=_LEG_BG, fg=_LEG_DIM, font=("TkDefaultFont", 7),
             anchor="w").pack(fill="x", padx=6, pady=(0, 6))

    def render():
        nonlocal tk_img, terrain_mode, particle_anim_running, oc_anim_running, display_scale_x, display_scale_y, _last_render_arr
        # Use simulation data if available and running
        use_sim_data = sim_state is not None and sim_running
        
        if mode_var.get() == "globe":
            day = int(sim_state.day_of_year) if (use_sim_data and sim_state is not None) else 80
            view_name = view_var.get()
            if terrain_mode == "loaded":
                elev_tex, _ = get_elevation_cache()
                if elev_tex is None:
                    terrain_mode = "procedural"
            # Wind Particles: hand off to animation loop (same as map mode)
            if view_name == "Wind Particles":
                if not particle_anim_running:
                    particle_anim_running = True
                    _update_wind_particles()
                return
            with log_time("Render globe"):
                if view_name in ("Biomes", "Cloud Cover", "Precipitation"):
                    # Compute equirectangular composite then project onto sphere
                    if use_sim_data and sim_state is not None and sim_state.elevation is not None:
                        tex = sim_state.elevation
                    elif terrain_mode == "loaded":
                        tex, _ = get_elevation_cache()
                        if tex is None:
                            tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
                    else:
                        tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
                    base_rgb = colorize(tex)
                    if view_name == "Biomes":
                        from climate_averages import KOPPEN_COLORS
                        if use_sim_data and sim_state is not None and sim_state.koppen_type is not None:
                            koppen = sim_state.koppen_type
                            biome_rgb = KOPPEN_COLORS[koppen]
                            alpha = (koppen > 0).astype(np.float32)
                        elif use_sim_data and sim_state is not None and sim_state.biome_type is not None:
                            biome = sim_state.biome_type
                            biome_colors = np.array([
                                [0.0, 0.0, 0.0], [0.9, 0.8, 0.5], [0.6, 0.8, 0.3],
                                [0.1, 0.5, 0.1], [0.7, 0.75, 0.8],
                            ], dtype=np.float32)
                            biome_rgb = biome_colors[biome]
                            alpha = (biome > 0).astype(np.float32)
                        else:
                            biome_rgb = None
                        if biome_rgb is not None:
                            comb = (1.0 - alpha[..., None]) * base_rgb + alpha[..., None] * biome_rgb
                        else:
                            comb = base_rgb
                    elif view_name == "Cloud Cover":
                        if use_sim_data and sim_state is not None and sim_state.cloud_cover is not None:
                            overlay, alpha = cloud_cover_to_rgb(sim_state.cloud_cover)
                            comb = (1.0 - alpha[..., None]) * base_rgb + alpha[..., None] * overlay
                        else:
                            comb = base_rgb
                    else:  # Precipitation -- cloud layer, then precip color on top (see map-mode
                           # branch above for the same treatment/rationale)
                        if use_sim_data and sim_state is not None and sim_state.cloud_cover is not None:
                            cloud_overlay, cloud_alpha = cloud_cover_to_rgb(sim_state.cloud_cover)
                            comb = (1.0 - cloud_alpha[..., None]) * base_rgb + cloud_alpha[..., None] * cloud_overlay
                        else:
                            comb = base_rgb
                        _precip_mode = time_scale_options.get(time_scale_var.get(), TimeScaleMode.DAILY)
                        _precip_use_avg = _precip_mode != TimeScaleMode.DAILY
                        if use_sim_data and sim_state is not None and _precip_use_avg and sim_state.climate_precip_avg is not None:
                            P = sim_state.climate_precip_avg.astype(np.float32)
                        elif use_sim_data and sim_state is not None and sim_state.precipitation is not None:
                            P = sim_state.precipitation.astype(np.float32)
                        else:
                            P, _, _ = generate_precipitation(*tex.shape, tex, day_of_year=int(day))
                        p_overlay, p_alpha = precipitation_to_rgb(P)
                        comb = (1.0 - p_alpha[..., None]) * comb + p_alpha[..., None] * p_overlay
                    # Match the horizontal flip that generate_sphere_image applies for loaded heightmaps
                    _, elev_key = get_elevation_cache()
                    if elev_key is not None and isinstance(elev_key, tuple) and len(elev_key) >= 1 and elev_key[0] == "loaded":
                        comb = np.fliplr(comb)
                    new_img = project_equirect_on_globe(np.clip(comb, 0.0, 1.0), size=size, radius=0.96, rot=(yaw, pitch, roll))
                else:
                    if view_name == "Wind Arrows":
                        view_name = "Wind"
                    if view_name == "Ocean Temperature":
                        view_name = "Temperature"
                    new_img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll), view=view_name, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get(), day_of_year=day)
            canvas.config(width=size, height=size)
        else:
            # Get elevation from simulation or generate
            if use_sim_data and sim_state.elevation is not None:
                tex = sim_state.elevation
            else:
                # Check terrain mode to determine how to get elevation
                if terrain_mode == "loaded":
                    # Use loaded heightmap from cache
                    elev_tex, elev_key = get_elevation_cache()
                    if elev_tex is not None:
                        tex = elev_tex
                    else:
                        # Fallback to procedural if cache is empty (shouldn't happen)
                        tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
                else:
                    # Procedural mode: check cache against parameters
                    params_key = (size, int(seed_var.get()), int(octaves_var.get()), float(freq_var.get()), float(lac_var.get()), float(gain_var.get()))
                    elev_tex, elev_key = get_elevation_cache()
                    if elev_tex is None or elev_key != params_key:
                        with log_time("Render map base (regen elevation)"):
                            tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
                    else:
                        tex = elev_tex
            if view_var.get() == "Wind Particles":
                if not particle_anim_running:
                    particle_anim_running = True
                    _update_wind_particles()
                return
            base_rgb = colorize(tex)
            
            if view_var.get() == "Temperature":
                if use_sim_data and sim_state.temperature is not None:
                    from temperature import temperature_to_rgb
                    # Show 2m air temperature; fall back to T_sst for old saves
                    T = sim_state.air_temperature if sim_state.air_temperature is not None else sim_state.temperature
                    overlay = temperature_to_rgb(T)
                else:
                    h, w = tex.shape
                    with log_time("Generate temperature overlay"):
                        overlay = generate_temperature_overlay(h, w, elevation=tex)
                alpha = 0.5
                comb = (1.0 - alpha) * base_rgb + alpha * overlay
                arr = (np.clip(comb, 0.0, 1.0) * 255).astype(np.uint8)
            elif view_var.get() == "Ocean Temperature":
                if use_sim_data and sim_state.temperature is not None:
                    from temperature import temperature_to_rgb
                    T = sim_state.temperature
                    ocean_rgb = temperature_to_rgb(T)
                else:
                    h, w = tex.shape
                    with log_time("Generate temperature overlay"):
                        ocean_rgb = generate_temperature_overlay(h, w, elevation=tex)
                ocean_mask, _ = get_masks(tex)
                base = np.zeros_like(ocean_rgb, dtype=np.float32)
                base[ocean_mask] = ocean_rgb[ocean_mask]
                if use_sim_data and sim_state.ice_cover is not None:
                    ice = np.clip(sim_state.ice_cover.astype(np.float32), 0.0, 1.0)
                    ice_mask = ocean_mask & (ice > 0.01)
                    base[ice_mask] = 1.0
                arr = (np.clip(base, 0.0, 1.0) * 255).astype(np.uint8)
            elif view_var.get() == "Precipitation":
                h, w = tex.shape
                day = int(sim_state.day_of_year) if (use_sim_data and sim_state is not None) else 80
                _precip_mode = time_scale_options.get(time_scale_var.get(), TimeScaleMode.DAILY)
                _precip_use_avg = _precip_mode != TimeScaleMode.DAILY
                if use_sim_data and _precip_use_avg and sim_state.climate_precip_avg is not None:
                    # Faster-than-DAILY speeds show the tracked climatological average
                    # rather than a single instantaneous (and, with storms enabled,
                    # noisy) snapshot.
                    P = sim_state.climate_precip_avg.astype(np.float32)
                elif use_sim_data and sim_state.precipitation is not None:
                    P = sim_state.precipitation.astype(np.float32)
                else:
                    T = sim_state.temperature if (use_sim_data and sim_state is not None) else None
                    u = sim_state.wind_u if (use_sim_data and sim_state is not None) else None
                    v = sim_state.wind_v if (use_sim_data and sim_state is not None) else None
                    pkey = (tex.shape, int(day), id(T), id(u), id(v))
                    if _PRECIP_VIEW_CACHE["key"] != pkey:
                        P, _, _ = generate_precipitation(
                            h,
                            w,
                            tex,
                            temperature=T,
                            wind_u=u,
                            wind_v=v,
                            day_of_year=int(day),
                        )
                        _PRECIP_VIEW_CACHE.update({"key": pkey, "P": P})
                    else:
                        P = _PRECIP_VIEW_CACHE["P"]
                # Cloud layer first (satellite-style gray/white texture), then precipitation
                # color on top where it's actually raining -- matches a real weather-radar
                # composite (clouds show the broader storm structure; precip color highlights
                # only the actively-precipitating cores within/around it).
                if use_sim_data and sim_state.cloud_cover is not None:
                    cloud_overlay, cloud_alpha = cloud_cover_to_rgb(sim_state.cloud_cover)
                    comb = (1.0 - cloud_alpha[..., None]) * base_rgb + cloud_alpha[..., None] * cloud_overlay
                else:
                    comb = base_rgb
                overlay, alpha = precipitation_to_rgb(P)
                comb = (1.0 - alpha[..., None]) * comb + alpha[..., None] * overlay
                arr = (np.clip(comb, 0.0, 1.0) * 255).astype(np.uint8)
            elif view_var.get() == "Biomes":
                # Biome visualization - prefer Köppen classification for detailed climate zones
                from climate_averages import KOPPEN_COLORS

                # Use Köppen if available, fall back to legacy biomes
                if use_sim_data and sim_state.koppen_type is not None:
                    # Köppen classification (20 climate types, updated every 30 days)
                    koppen = sim_state.koppen_type
                    biome_rgb = KOPPEN_COLORS[koppen]
                    alpha = (koppen > 0).astype(np.float32)  # 0 = ocean
                elif use_sim_data and sim_state.biome_type is not None:
                    # Legacy biome classification (5 types)
                    biome = sim_state.biome_type
                    biome_colors = np.array([
                        [0.0, 0.0, 0.0],    # Ocean
                        [0.9, 0.8, 0.5],    # Desert
                        [0.6, 0.8, 0.3],    # Grassland
                        [0.1, 0.5, 0.1],    # Forest
                        [0.7, 0.75, 0.8],   # Tundra
                    ], dtype=np.float32)
                    biome_rgb = biome_colors[biome]
                    alpha = (biome > 0).astype(np.float32)
                elif use_sim_data and sim_state.temperature is not None and sim_state.precipitation is not None:
                    # Fallback: compute from instantaneous values
                    from carbon_cycle import compute_biome_type
                    land_mask = (tex > 0.02).astype(np.float32)
                    biome = compute_biome_type(sim_state.temperature, sim_state.precipitation, land_mask)
                    biome_colors = np.array([
                        [0.0, 0.0, 0.0], [0.9, 0.8, 0.5], [0.6, 0.8, 0.3],
                        [0.1, 0.5, 0.1], [0.7, 0.75, 0.8],
                    ], dtype=np.float32)
                    biome_rgb = biome_colors[biome]
                    alpha = (biome > 0).astype(np.float32)
                else:
                    biome_rgb = None
                    alpha = None

                if biome_rgb is not None:
                    # Blend with terrain (ocean uses base, land uses biome/Köppen color)
                    comb = (1.0 - alpha[..., None]) * base_rgb + alpha[..., None] * biome_rgb
                    arr = (np.clip(comb, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    arr = (base_rgb * 255).astype(np.uint8)
            elif view_var.get() == "Cloud Cover":
                if use_sim_data and sim_state.cloud_cover is not None:
                    overlay, alpha = cloud_cover_to_rgb(sim_state.cloud_cover)
                    comb = (1.0 - alpha[..., None]) * base_rgb + alpha[..., None] * overlay
                    arr = (np.clip(comb, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    arr = (base_rgb * 255).astype(np.uint8)
            elif view_var.get() == "Wind Arrows":
                if use_sim_data and sim_state.wind_u is not None and sim_state.wind_v is not None:
                    u, v = sim_state.wind_u, sim_state.wind_v
                else:
                    wkey = (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()), "bilinear")
                    if _WIND_CACHE["key"] != wkey:
                        with log_time("Generate wind field"):
                            u, v = generate_wind_field(*tex.shape, elevation=tex, upsample="bilinear", weather_amp=0.35, debug_log=False)
                            _WIND_CACHE.update({"key": wkey, "u": u, "v": v})
                    else:
                        u, v = _WIND_CACHE["u"], _WIND_CACHE["v"]
                speed = np.hypot(u, v).astype(np.float32)
                base_rgb = np.clip(0.60 * colorize(tex) + 0.40 * wind_speed_to_rgb(speed), 0.0, 1.0)
                arrows = render_wind_arrows(*tex.shape, u, v, target_arrows=int(wind_arrows_var.get()), scale=float(wind_scale_var.get()))
                arr = (np.clip(base_rgb + arrows, 0.0, 1.0) * 255).astype(np.uint8)
            elif view_var.get() == "Ocean Currents":
                if not oc_anim_running:
                    oc_anim_running = True
                    _update_ocean_particles()
                return
            else:
                arr = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            _last_render_arr = arr
            new_img = Image.fromarray(arr)
            h, w = tex.shape
            canvas.config(width=w, height=h)
        # Scale rendered image to fill the current canvas allocation (window resize support)
        _alloc_w = canvas.winfo_width()
        _alloc_h = canvas.winfo_height()
        _img_w, _img_h = new_img.size
        if _alloc_w > 1 and _alloc_h > 1:
            _scale = min(_alloc_w / _img_w, _alloc_h / _img_h)
            if abs(_scale - 1.0) > 0.005:
                _dw = max(1, int(_img_w * _scale))
                _dh = max(1, int(_img_h * _scale))
                new_img = new_img.resize((_dw, _dh), Image.NEAREST)
            display_scale_x = new_img.width / _img_w
            display_scale_y = new_img.height / _img_h
        else:
            display_scale_x = 1.0
            display_scale_y = 1.0
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)
        # Update status with simulation day (only if not paused)
        if use_sim_data and not sim_paused:
            sim_status_var.set(f"Day: {sim_state.day_of_year:.0f}")
        # Clear lat/lon when switching out of map
        if mode_var.get() != "map":
            latlon_var.set("")

    def start_simulation():
        nonlocal sim_state, sim_thread, sim_running, sim_paused, _sim_ever_started
        if not _sim_ever_started:
            # On first Start: load the active file's autosave if enabled and available
            if _auto_save_enabled and _current_state_path.exists():
                try:
                    sim_state = load_state(_current_state_path)
                    total_years = sim_state.total_days / 365.2422
                    sim_status_var.set(f"Loaded Y{total_years:.1f}")
                    LOG.info(f"Autosave loaded: day {sim_state.total_days:.0f} ({total_years:.2f} years)")
                except Exception as e:
                    LOG.warning(f"Failed to load autosave ({e}); starting fresh.")
        _sim_ever_started = True

        # Start or resume simulation thread
        if sim_thread is None or not sim_thread.is_alive():
            sim_thread = SimulationThread(
                sim_state,
                days_per_step=sim_speed,
                wind_block_size=int(wind_block_size_var.get()),
                diagnostics=diagnostics,
                time_scale_mode=time_scale_options.get(time_scale_var.get(), TimeScaleMode.DAILY),
            )
            sim_thread.start()

        sim_thread.resume()
        sim_running = True
        sim_paused = False
        sim_status_var.set("Running")
        year = int(sim_state.total_days // 365.2422) + 1
        month = int((sim_state.day_of_year / 365.2422) * 12) + 1
        sim_cycle_var.set(f"Y{year} M{month}")
        # If particle view is selected, start animation loop (both map and globe).
        nonlocal particle_anim_running
        if view_var.get() == "Wind Particles" and not particle_anim_running:
            particle_anim_running = True
            _update_wind_particles()
        # Start UI update loop
        update_from_simulation()
    
    def stop_simulation():
        nonlocal sim_thread, sim_running, sim_paused
        if sim_thread:
            sim_thread.pause()
        sim_running = False
        sim_paused = False
        sim_status_var.set("Stopped")
        cycle_progress_var.set(0.0)
        cycle_detail_var.set("")
    
    def pause_simulation():
        nonlocal sim_thread, sim_paused
        if sim_running:
            sim_paused = not sim_paused
            if sim_thread:
                if sim_paused:
                    sim_thread.pause()
                else:
                    sim_thread.resume()
            sim_status_var.set("Paused" if sim_paused else "Running")
    
    def reset_simulation():
        nonlocal sim_state, sim_thread, sim_running, sim_paused
        # Stop the simulation thread if running
        if sim_thread:
            sim_thread.stop()
            sim_thread = None
        # Reset to a freshly initialized (stopped) state from current elevation.
        use_sim_data = sim_state is not None and sim_running
        if use_sim_data and sim_state is not None and sim_state.elevation is not None:
            tex = sim_state.elevation
        else:
            tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
        _init_sim_state_from_elevation(tex)
        cycle_progress_var.set(0.0)
        cycle_detail_var.set("")
        render()

    def _update_zoom_preview(cx: int, cy: int) -> None:
        """Display magnified patch of the rendered map centered on cursor cell (cx, cy)."""
        nonlocal _zoom_tk_img
        if _last_render_arr is None:
            return
        harr, warr = _last_render_arr.shape[:2]
        half = _ZOOM_CELLS // 2
        y0 = max(0, cy - half)
        y1 = min(harr, cy + half + 1)
        x0 = max(0, cx - half)
        x1 = min(warr, cx + half + 1)
        patch = _last_render_arr[y0:y1, x0:x1]
        # Pad to full _ZOOM_CELLS × _ZOOM_CELLS (black border near map edges)
        pad_arr = np.zeros((_ZOOM_CELLS, _ZOOM_CELLS, 3), dtype=np.uint8)
        dy0 = half - (cy - y0)
        dx0 = half - (cx - x0)
        pad_arr[dy0:dy0 + (y1 - y0), dx0:dx0 + (x1 - x0)] = patch
        zoom_px = _ZOOM_CELLS * _ZOOM_FACTOR
        zoom_img = Image.fromarray(pad_arr).resize((zoom_px, zoom_px), Image.NEAREST)
        # Yellow crosshair at the center cell
        draw = ImageDraw.Draw(zoom_img)
        mid = half * _ZOOM_FACTOR + _ZOOM_FACTOR // 2
        arm = _ZOOM_FACTOR + 3
        draw.line([(mid - arm, mid), (mid + arm, mid)], fill=(255, 255, 0), width=1)
        draw.line([(mid, mid - arm), (mid, mid + arm)], fill=(255, 255, 0), width=1)
        _zoom_canvas.config(width=zoom_px, height=zoom_px)
        _zoom_tk_img = ImageTk.PhotoImage(zoom_img)
        _zoom_canvas.itemconfig(_zoom_img_id, image=_zoom_tk_img)

    def update_cursor_display(x: int, y: int):
        """Update the cursor display and tooltip for given canvas coordinates."""
        if mode_var.get() != "map":
            _tooltip_win.withdraw()
            return
        use_sim_data = sim_state is not None and sim_running
        if use_sim_data and sim_state.elevation is not None:
            tex = sim_state.elevation
        else:
            tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
        h, w = tex.shape
        # Convert canvas (display) coords to simulation grid coords
        x = int(x / display_scale_x)
        y = int(y / display_scale_y)
        if 0 <= x < w and 0 <= y < h:
            lon = (x / w) * 360.0 - 180.0
            lat = 90.0 - (y / h) * 180.0
            elev_raw = float(tex[int(y), int(x)])
            if terrain_mode == "loaded":
                pixel_display = f"norm {elev_raw:.3f}"
                if elev_raw == 0.0:
                    alt_m = 0.0
                elif elev_raw <= 0.03:
                    alt_m = (elev_raw / 0.03) * 100.0
                else:
                    normalized = (elev_raw - 0.03) / 0.97
                    alt_m = 100.0 + (normalized ** 2.5) * 8748.0
                is_ocean = (elev_raw == 0.0)
            else:
                pixel_display = ""
                sea_level = 0.2
                if elev_raw <= sea_level:
                    alt_m = 0.0
                else:
                    elevation_above_sea = elev_raw - sea_level
                    alt_m = (elevation_above_sea / (1.0 - sea_level)) ** 2.0 * 8848.0
                is_ocean = (elev_raw <= sea_level)

            lat_str = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
            lon_str = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
            hdr = f"{lat_str}  {lon_str}"
            px_str = f", {pixel_display}" if pixel_display else ""
            view = view_var.get()

            if view in ("Wind Arrows", "Wind Particles"):
                if use_sim_data and sim_state.wind_u is not None and sim_state.wind_v is not None:
                    u, v = sim_state.wind_u, sim_state.wind_v
                elif _WIND_CACHE["u"] is None or _WIND_CACHE["u"].shape != (h, w):
                    u, v = generate_wind_field(h, w, elevation=tex, upsample="bilinear", weather_amp=0.35, debug_log=False)
                    _WIND_CACHE.update({"key": (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()), "bilinear"), "u": u, "v": v})
                else:
                    u, v = _WIND_CACHE["u"], _WIND_CACHE["v"]
                u_val = float(u[int(y), int(x)])
                v_val = float(v[int(y), int(x)])
                speed = float(np.hypot(u_val, v_val))
                angle = float(np.degrees(np.arctan2(v_val, u_val))) % 360
                dirs = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
                dir_str = dirs[int((angle + 22.5) / 45) % 8]
                latlon_var.set(f"lat {lat:6.2f}°, lon {lon:7.2f}°{px_str}, elev {alt_m:5.0f}m, wind {speed:5.1f} m/s")
                tt_lines = [hdr, f"Elev:  {alt_m:,.0f} m", f"Wind:  {speed:.1f} m/s {dir_str}"]

            elif view == "Ocean Currents":
                day = int(sim_state.day_of_year) if (use_sim_data and sim_state is not None) else 80
                tdays = float(sim_state.total_days) if (use_sim_data and sim_state is not None) else float(day)
                ckey = (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()), "ocean_currents", int(tdays))
                if _OCEAN_CURRENT_CACHE["key"] != ckey:
                    wu = sim_state.wind_u if (use_sim_data and sim_state is not None) else None
                    wv = sim_state.wind_v if (use_sim_data and sim_state is not None) else None
                    u, v = generate_ocean_currents(tex, wind_u=wu, wind_v=wv, day_of_year=day, time_days=tdays)
                    _OCEAN_CURRENT_CACHE.update({"key": ckey, "u": u, "v": v})
                else:
                    u, v = _OCEAN_CURRENT_CACHE["u"], _OCEAN_CURRENT_CACHE["v"]
                speed = float(np.hypot(u[int(y), int(x)], v[int(y), int(x)]))
                latlon_var.set(f"lat {lat:6.2f}°, lon {lon:7.2f}°{px_str}, elev {alt_m:5.0f}m, current {speed:5.2f} m/s")
                tt_lines = [hdr, f"Current: {speed:.3f} m/s"]

            elif view == "Temperature":
                if use_sim_data and sim_state.temperature is not None:
                    _T_air = sim_state.air_temperature if sim_state.air_temperature is not None else sim_state.temperature
                    T_air_c = float(_T_air[int(y), int(x)]) - 273.15
                    T_sst_c = float(sim_state.temperature[int(y), int(x)]) - 273.15
                else:
                    T_air_c = temperature_kelvin_for_lat(np.deg2rad(lat)) - 273.15
                    T_sst_c = None
                latlon_var.set(f"lat {lat:6.2f}°, lon {lon:7.2f}°{px_str}, elev {alt_m:5.0f}m, T {T_air_c:6.1f}°C")
                tt_lines = [hdr, f"Elev:    {alt_m:,.0f} m", f"Air T:   {T_air_c:.1f}°C"]
                if T_sst_c is not None and is_ocean:
                    tt_lines.append(f"SST:     {T_sst_c:.1f}°C")

            elif view == "Ocean Temperature":
                if use_sim_data and sim_state.temperature is not None:
                    T_sst_c = float(sim_state.temperature[int(y), int(x)]) - 273.15
                else:
                    T_sst_c = temperature_kelvin_for_lat(np.deg2rad(lat)) - 273.15
                latlon_var.set(f"lat {lat:6.2f}°, lon {lon:7.2f}°{px_str}, SST {T_sst_c:.1f}°C")
                tt_lines = [hdr, f"SST:     {T_sst_c:.1f}°C"]
                if use_sim_data and sim_state.ice_cover is not None:
                    ice = float(sim_state.ice_cover[int(y), int(x)])
                    if ice > 0.01:
                        tt_lines.append(f"Ice:     {ice * 100:.0f}%")

            elif view == "Precipitation":
                _tt_precip_mode = time_scale_options.get(time_scale_var.get(), TimeScaleMode.DAILY)
                _tt_precip_use_avg = _tt_precip_mode != TimeScaleMode.DAILY
                if use_sim_data and _tt_precip_use_avg and sim_state.climate_precip_avg is not None:
                    precip = float(sim_state.climate_precip_avg[int(y), int(x)])
                elif use_sim_data and sim_state.precipitation is not None:
                    precip = float(sim_state.precipitation[int(y), int(x)])
                else:
                    precip = 0.0
                    _tt_precip_use_avg = False
                if use_sim_data and sim_state.temperature is not None:
                    _T_disp = sim_state.air_temperature if sim_state.air_temperature is not None else sim_state.temperature
                    T_celsius = float(_T_disp[int(y), int(x)]) - 273.15
                else:
                    T_celsius = temperature_kelvin_for_lat(np.deg2rad(lat)) - 273.15
                latlon_var.set(f"lat {lat:6.2f}°, lon {lon:7.2f}°{px_str}, elev {alt_m:5.0f}m, T {T_celsius:.1f}°C")
                _precip_label = "Precip (avg):" if _tt_precip_use_avg else "Precip:"
                tt_lines = [hdr, f"{_precip_label} {precip:.3f} mm/day", f"Air T:   {T_celsius:.1f}°C"]
                if use_sim_data and sim_state.cloud_cover is not None:
                    cloud = float(sim_state.cloud_cover[int(y), int(x)])
                    tt_lines.append(f"Cloud:   {cloud * 100:.0f}%")

            elif view == "Biomes":
                biome_name = "Ocean" if is_ocean else "Land"
                if use_sim_data and sim_state.koppen_type is not None:
                    kt = int(sim_state.koppen_type[int(y), int(x)])
                    biome_name = _KNAMES.get(kt, f"Type {kt}") if kt > 0 else "Ocean"
                elif use_sim_data and sim_state.biome_type is not None:
                    bt = int(sim_state.biome_type[int(y), int(x)])
                    legacy = {0: "Ocean", 1: "Desert", 2: "Grassland", 3: "Forest", 4: "Tundra"}
                    biome_name = legacy.get(bt, f"Biome {bt}")
                if use_sim_data and sim_state.temperature is not None:
                    _T_disp = sim_state.air_temperature if sim_state.air_temperature is not None else sim_state.temperature
                    T_celsius = float(_T_disp[int(y), int(x)]) - 273.15
                else:
                    T_celsius = temperature_kelvin_for_lat(np.deg2rad(lat)) - 273.15
                latlon_var.set(f"lat {lat:6.2f}°, lon {lon:7.2f}°{px_str}, {biome_name}")
                tt_lines = [hdr, biome_name, f"Air T:   {T_celsius:.1f}°C"]

            elif view == "Cloud Cover":
                if use_sim_data and sim_state.cloud_cover is not None:
                    cloud = float(sim_state.cloud_cover[int(y), int(x)])
                else:
                    cloud = 0.0
                if use_sim_data and sim_state.temperature is not None:
                    _T_disp = sim_state.air_temperature if sim_state.air_temperature is not None else sim_state.temperature
                    T_celsius = float(_T_disp[int(y), int(x)]) - 273.15
                else:
                    T_celsius = temperature_kelvin_for_lat(np.deg2rad(lat)) - 273.15
                latlon_var.set(f"lat {lat:6.2f}°, lon {lon:7.2f}°{px_str}, clouds {cloud * 100:.0f}%, T {T_celsius:.1f}°C")
                tt_lines = [hdr, f"Clouds:  {cloud * 100:.0f}%", f"Air T:   {T_celsius:.1f}°C"]

            else:  # Terrain and fallback
                if use_sim_data and sim_state.temperature is not None:
                    _T_disp = sim_state.air_temperature if sim_state.air_temperature is not None else sim_state.temperature
                    T_kelvin = float(_T_disp[int(y), int(x)])
                else:
                    T_kelvin = temperature_kelvin_for_lat(np.deg2rad(lat))
                T_celsius = T_kelvin - 273.15
                latlon_var.set(f"lat {lat:6.2f}°, lon {lon:7.2f}°{px_str}, elev {alt_m:5.0f}m, T {T_celsius:6.1f}°C")
                terrain_type = "Ocean" if is_ocean else "Land"
                tt_lines = [hdr, terrain_type, f"Elev:  {alt_m:,.0f} m", f"Air T: {T_celsius:.1f}°C"]

            _update_zoom_preview(int(x), int(y))
            _tooltip_var.set("\n".join(tt_lines))
        else:
            latlon_var.set("")
            _tooltip_var.set("")
            _tooltip_win.withdraw()
    
    def update_from_simulation():
        """Pull latest state from simulation thread and update UI (called by timer)."""
        nonlocal sim_state, sim_thread

        if sim_thread:
            try:
                # Non-blocking get - pull latest state if available
                new_state = sim_thread.state_queue.get_nowait()
                sim_state = new_state

                # Also try to get temperature components (for diagnostics)
                try:
                    temp_components = sim_thread.component_queue.get_nowait()
                except Empty:
                    temp_components = None

                # Update UI with new state
                year = int(sim_state.total_days // 365.2422) + 1
                month = int((sim_state.day_of_year / 365.2422) * 12) + 1
                sim_cycle_var.set(f"Y{year} M{month}")
                if not sim_paused:
                    sim_status_var.set(f"Day: {sim_state.day_of_year:.0f}")

                # Update year-progress bar
                _pct = (sim_state.day_of_year / 365.2422) * 100.0
                cycle_progress_var.set(_pct)
                _mn = _MONTH_NAMES[(month - 1) % 12]
                cycle_detail_var.set(f"Day {sim_state.day_of_year:>3.0f}/365  {_mn}  ({_pct:>5.1f}%)")

                # Update display
                render()

                # Update cursor display at last known mouse position
                if mode_var.get() == "map":
                    update_cursor_display(last_mouse_pos[0], last_mouse_pos[1])

            except Empty:
                # No new state available - that's okay, just skip this update
                pass

        # Schedule next UI update (50ms = ~20 FPS for UI responsiveness)
        if sim_running or sim_thread:
            root.after(50, update_from_simulation)
    
    def on_motion(e):
        nonlocal last_mouse_pos
        last_mouse_pos = (e.x, e.y)
        update_cursor_display(e.x, e.y)
        if mode_var.get() == "map" and _tooltip_var.get():
            rx = canvas.winfo_rootx() + e.x + 16
            ry = canvas.winfo_rooty() + e.y + 16
            _tooltip_win.wm_geometry(f"+{rx}+{ry}")
            _tooltip_win.deiconify()
            _tooltip_win.lift()
        else:
            _tooltip_win.withdraw()

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

    def _on_view_change(*_):
        render()
        # Show the legend only in Biomes map view; hide it everywhere else
        if view_var.get() == "Biomes" and mode_var.get() == "map":
            legend_outer.pack(side="left", anchor="nw", padx=(4, 4), pady=4)
        else:
            legend_outer.pack_forget()

    mode_var.trace_add("write", _on_view_change)
    view_var.trace_add("write", _on_view_change)
    def on_close():
        nonlocal sim_thread
        # Stop simulation thread if running
        if sim_thread:
            sim_thread.stop()
            sim_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
        # Persist current UI parameters to settings.json
        # Autosave simulation state (to the active file) if enabled
        if _auto_save_enabled and sim_state is not None:
            try:
                save_state(sim_state, _current_state_path)
                LOG.info(f"Autosave written to {_current_state_path.name}: day {sim_state.total_days:.0f}")
            except Exception as e:
                LOG.warning(f"Autosave failed: {e}")

        s = {
            "seed": int(seed_var.get()),
            "octaves": int(octaves_var.get()),
            "freq": float(freq_var.get()),
            "lac": float(lac_var.get()),
            "gain": float(gain_var.get()),
            "wind_arrows": int(wind_arrows_var.get()),
            "wind_scale": float(wind_scale_var.get()),
            "wind_block_size": int(wind_block_size_var.get()),
            "auto_save_state": bool(_auto_save_enabled),
            "last_state_path": str(_current_state_path),
        }
        save_settings(s)
        graphs_controller.close()
        root.destroy()

    # --- Unified menu bar (built last, once every command it references exists) ---
    menubar = tk.Menu(root)

    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Open Heightmap...", command=load_heightmap)
    file_menu.add_command(label="Use Procedural Terrain", command=use_procedural_terrain)
    file_menu.add_separator()
    file_menu.add_command(label="Open State...", command=_do_load_state, accelerator="Ctrl+O")
    file_menu.add_command(label="Save State", command=_do_save_state, accelerator="Ctrl+S")
    file_menu.add_command(label="Save State As...", command=_do_save_state_as, accelerator="Ctrl+Shift+S")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=on_close)
    menubar.add_cascade(label="File", menu=file_menu)

    simulation_menu = tk.Menu(menubar, tearoff=0)
    simulation_menu.add_command(label="Export Data...", command=export_data)
    simulation_menu.add_command(label="Run Benchmark", command=run_benchmark)
    menubar.add_cascade(label="Simulation", menu=simulation_menu)

    terrain_menu = tk.Menu(menubar, tearoff=0)
    terrain_menu.add_command(label="Regenerate Terrain", command=do_regen)
    menubar.add_cascade(label="Terrain", menu=terrain_menu)

    root.config(menu=menubar)
    root.bind("<Control-o>", lambda e: _do_load_state())
    root.bind("<Control-s>", lambda e: _do_save_state())
    root.bind("<Control-Shift-S>", lambda e: _do_save_state_as())

    def _on_window_resize(event):
        nonlocal _resize_job
        if event.widget is root:
            if _resize_job is not None:
                root.after_cancel(_resize_job)
            _resize_job = root.after(150, render)

    root.bind("<Configure>", _on_window_resize)
    root.bind("<Escape>", lambda e: on_close())
    root.protocol("WM_DELETE_WINDOW", on_close)
    canvas.bind("<Motion>", on_motion)
    canvas.bind("<Leave>", lambda e: _tooltip_win.withdraw())

    def on_wheel(e):
        nonlocal _ZOOM_CELLS
        delta = -1 if (e.delta > 0 or e.num == 4) else 1  # scroll up = zoom in (fewer cells)
        _ZOOM_CELLS = max(5, min(51, _ZOOM_CELLS + delta * 2))
        if mode_var.get() == "map" and _tooltip_var.get():
            _update_zoom_preview(last_mouse_pos[0], last_mouse_pos[1])

    canvas.bind("<MouseWheel>", on_wheel)   # Windows / macOS
    canvas.bind("<Button-4>", on_wheel)     # Linux scroll up
    canvas.bind("<Button-5>", on_wheel)     # Linux scroll down
    root.mainloop()


if __name__ == "__main__":
    main()

