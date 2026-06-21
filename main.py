"""Main entry point for planet simulator.

Launches the GUI application for viewing and interacting with the planet simulation.
All modules are kept separated: terrain, atmosphere, temperature, and simulate.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
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
from terrain import precipitation_to_rgb
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
    root.resizable(False, False)

    # Simulation state
    sim_state: PlanetState | None = None
    sim_thread: SimulationThread | None = None
    sim_running = False
    sim_paused = False
    sim_speed = 1.0  # days per step
    last_mouse_pos = (0, 0)  # Track last mouse position for cursor updates
    _sim_ever_started = False  # False until user clicks Start for the first time
    
    # Terrain mode: "procedural" or "loaded"
    terrain_mode = "procedural"
    loaded_heightmap_path = None
    
    # Controls
    mode_var = tk.StringVar(value="map")
    view_var = tk.StringVar(value="Terrain")
    latlon_var = tk.StringVar(value="")
    controls = tk.Frame(root)
    controls.pack(fill="x")
    tk.Radiobutton(controls, text="Globe", variable=mode_var, value="globe").pack(side="left")
    tk.Radiobutton(controls, text="Map", variable=mode_var, value="map").pack(side="left")
    tk.Label(controls, text="View").pack(side="left", padx=(8,0))
    tk.OptionMenu(
        controls,
        view_var,
        "Terrain",
        "Temperature",
        "Ocean Temperature",
        "Precipitation",
        "Biomes",
        "Wind Arrows",
        "Ocean Currents",
        "Wind Particles",
        "Cloud Cover",
    ).pack(side="left")
    
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

    def _wind_uv_for_display() -> tuple[np.ndarray, np.ndarray] | None:
        """Get (u,v) from sim_state for wind particle animation."""
        nonlocal sim_state
        if sim_state is None or sim_state.wind_u is None or sim_state.wind_v is None:
            return None
        return sim_state.wind_u, sim_state.wind_v

    def _update_wind_particles() -> None:
        """Animate wind particles in map mode using a fading trail buffer."""
        nonlocal tk_img, particle_xy, particle_age, trail, base_wind_rgb_u8, last_wind_key, last_anim_t, particle_anim_running
        if mode_var.get() != "map" or view_var.get() != "Wind Particles":
            particle_anim_running = False
            return
        # Only animate while the simulation is running (Start). Otherwise, show a static frame.
        if not sim_running or sim_paused:
            uv = _wind_uv_for_display()
            if uv is None:
                return
            u, v = uv
            speed = np.hypot(u, v).astype(np.float32)
            base_rgb = wind_speed_to_rgb(speed)
            out = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            new_img = Image.fromarray(out)
            tk_img = ImageTk.PhotoImage(new_img)
            canvas.itemconfig(img_id, image=tk_img)
            canvas.config(width=out.shape[1], height=out.shape[0])
            particle_anim_running = False
            return

        # Use simulation elevation for dimensions (required for consistent indexing).
        if sim_state is None or sim_state.elevation is None:
            return
        tex = sim_state.elevation
        h, w = tex.shape
        if trail is None or trail.shape != (h, w) or particle_xy is None or particle_age is None:
            # Particle count: tie to "Arrows" control, but scale up for particles
            _init_particles(h, w, n=int(wind_arrows_var.get()) * 2)

        uv = _wind_uv_for_display()
        if uv is None:
            return
        u, v = uv
        # Cache base wind speed colormap, recompute only if u/v identity changes
        uv_key = (id(u), id(v), h, w)
        if base_wind_rgb_u8 is None or last_wind_key != uv_key:
            speed = np.hypot(u, v).astype(np.float32)
            base_rgb = wind_speed_to_rgb(speed)  # absolute scaling now
            base_wind_rgb_u8 = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8)
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

        new_img = Image.fromarray(out)
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)
        canvas.config(width=w, height=h)

        # Schedule next frame
        root.after(50, _update_wind_particles)
    
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

    tk.Button(controls, text="Export Data", command=export_data).pack(side="right", padx=10)
    tk.Button(controls, text="Benchmark", command=run_benchmark).pack(side="right", padx=10)

    # --- State save/load controls ---
    auto_save_var = tk.BooleanVar(value=_auto_save_enabled)
    save_info_var = tk.StringVar(value="")

    def _refresh_save_info() -> None:
        if AUTOSAVE_PATH.exists():
            size_kb = AUTOSAVE_PATH.stat().st_size / 1024
            save_info_var.set(f"Save: {size_kb:.0f} KB")
        else:
            save_info_var.set("No save")

    def _do_save_state() -> None:
        nonlocal sim_state
        if sim_state is None:
            messagebox.showinfo("Save State", "No simulation state to save. Start the simulation first.")
            return
        try:
            save_state(sim_state, AUTOSAVE_PATH)
            _refresh_save_info()
            total_years = sim_state.total_days / 365.2422
            messagebox.showinfo("Save State", f"State saved.\nSimulation day {sim_state.total_days:.0f} ({total_years:.2f} years)")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _do_clear_save() -> None:
        if AUTOSAVE_PATH.exists():
            AUTOSAVE_PATH.unlink()
        _refresh_save_info()
        messagebox.showinfo("Clear Save", "Saved state cleared. Next start will use fresh initial conditions.")

    def _do_load_state() -> None:
        nonlocal sim_state, sim_running, sim_paused, sim_thread
        if not AUTOSAVE_PATH.exists():
            messagebox.showinfo("Load State", "No saved state found.\nUse 'Save State Now' to create one first.")
            return
        # Stop and discard the thread so Start recreates it from the loaded state
        if sim_thread and sim_thread.is_alive():
            sim_thread.stop()
            sim_thread = None
        sim_running = False
        sim_paused = False
        try:
            sim_state = load_state(AUTOSAVE_PATH)
            total_years = sim_state.total_days / 365.2422
            sim_status_var.set("Stopped")
            year = int(sim_state.total_days // 365.2422) + 1
            month = int((sim_state.day_of_year / 365.2422) * 12) + 1
            sim_cycle_var.set(f"Y{year} M{month}")
            _refresh_save_info()
            render()
            messagebox.showinfo("Load State", f"State loaded.\nSimulation day {sim_state.total_days:.0f} ({total_years:.2f} years)")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _on_auto_save_toggle() -> None:
        nonlocal _auto_save_enabled
        _auto_save_enabled = auto_save_var.get()

    save_row = tk.Frame(root)
    save_row.pack(fill="x")
    tk.Checkbutton(save_row, text="Auto-save/load state", variable=auto_save_var, command=_on_auto_save_toggle).pack(side="left", padx=4)
    tk.Button(save_row, text="Save State Now", command=_do_save_state).pack(side="left", padx=4)
    tk.Button(save_row, text="Load State", command=_do_load_state).pack(side="left", padx=4)
    tk.Button(save_row, text="Clear Saved State", command=_do_clear_save).pack(side="left", padx=4)
    tk.Label(save_row, textvariable=save_info_var, anchor="w", fg="gray").pack(side="left", padx=8)
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
    
    # Simulation controls
    sim_controls = tk.Frame(root)
    sim_controls.pack(fill="x")
    sim_status_var = tk.StringVar(value="Stopped")
    sim_cycle_var = tk.StringVar(value="Year: 1")
    tk.Label(sim_controls, text="Simulation:").pack(side="left", padx=(4,0))
    sim_status_label = tk.Label(sim_controls, textvariable=sim_status_var, width=12, anchor="w")
    sim_status_label.pack(side="left", padx=4)
    sim_cycle_label = tk.Label(sim_controls, textvariable=sim_cycle_var, width=12, anchor="w")
    sim_cycle_label.pack(side="left", padx=8)
    tk.Button(sim_controls, text="Start", command=lambda: start_simulation()).pack(side="left", padx=2)
    tk.Button(sim_controls, text="Stop", command=lambda: stop_simulation()).pack(side="left", padx=2)
    tk.Button(sim_controls, text="Pause", command=lambda: pause_simulation()).pack(side="left", padx=2)
    tk.Button(sim_controls, text="Reset", command=lambda: reset_simulation()).pack(side="left", padx=2)
    def on_graphs_toggle():
        graphs_controller.set_enabled(graphs_enabled_var.get())
    tk.Checkbutton(sim_controls, text="Graphs", variable=graphs_enabled_var, command=on_graphs_toggle).pack(side="left", padx=6)

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
    tk.Label(sim_controls, text="Speed:").pack(side="left", padx=(12, 0))
    tk.OptionMenu(sim_controls, time_scale_var, *time_scale_options.keys()).pack(side="left", padx=2)

    # Wind controls
    wind_arrows_var = tk.IntVar(value=int(settings.get("wind_arrows", default_settings["wind_arrows"])))
    wind_scale_var = tk.DoubleVar(value=float(settings.get("wind_scale", default_settings["wind_scale"])))
    wind_block_size_var = tk.IntVar(value=int(settings.get("wind_block_size", default_settings["wind_block_size"])))
    # Precipitation simulation removed.
    def add_wind_controls():
        frm = tk.Frame(root)
        frm.pack(fill="x")
        def add(parent, label, var, width=6):
            f = tk.Frame(parent); f.pack(side="left", padx=4); tk.Label(f, text=label).pack(side="left"); tk.Entry(f, textvariable=var, width=width).pack(side="left")
        add(frm, "Arrows", wind_arrows_var)
        add(frm, "Scale", wind_scale_var)
        add(frm, "WindBS", wind_block_size_var, width=4)
        return frm
    wind_controls = add_wind_controls()

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
        entry = tk.Entry(frm, textvariable=var, width=width)
        entry.pack(side="left")
        return entry

    seed_entry = add_labeled_entry(controls, "Seed", seed_var)
    octaves_entry = add_labeled_entry(controls, "Oct", octaves_var)
    freq_entry = add_labeled_entry(controls, "Freq", freq_var)
    lac_entry = add_labeled_entry(controls, "Lac", lac_var)
    gain_entry = add_labeled_entry(controls, "Gain", gain_var)

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
    
    # File menu
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Heightmap...", command=load_heightmap)
    file_menu.add_command(label="Use Procedural Terrain", command=use_procedural_terrain)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)

    def do_regen():
        nonlocal tk_img, terrain_mode
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
                base_rgb = wind_speed_to_rgb(speed)
                arrows = render_wind_arrows(*tex.shape, u, v, target_arrows=int(wind_arrows_var.get()), scale=float(wind_scale_var.get()))
                comb = np.clip(base_rgb + arrows, 0.0, 1.0)
                arr = (comb * 255).astype(np.uint8)
            elif view_var.get() == "Ocean Currents":
                ckey = (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()), "ocean_currents")
                if _OCEAN_CURRENT_CACHE["key"] != ckey:
                    # Phase 3: Pass wind fields for Ekman transport coupling
                    wu = sim_state.wind_u if (sim_state is not None and sim_running) else None
                    wv = sim_state.wind_v if (sim_state is not None and sim_running) else None
                    u, v = generate_ocean_currents(tex, wind_u=wu, wind_v=wv, day_of_year=day, time_days=float(day))
                    _OCEAN_CURRENT_CACHE.update({"key": ckey, "u": u, "v": v})
                else:
                    u, v = _OCEAN_CURRENT_CACHE["u"], _OCEAN_CURRENT_CACHE["v"]
                speed = np.hypot(u, v).astype(np.float32)
                base_rgb = wind_speed_to_rgb(speed)
                arrows = render_wind_arrows(*tex.shape, u, v, target_arrows=int(wind_arrows_var.get()), scale=float(wind_scale_var.get()))
                ocean_mask, _ = get_masks(tex)
                mask3 = ocean_mask[..., None].astype(np.float32)
                comb = np.clip((base_rgb + arrows) * mask3, 0.0, 1.0)
                arr = (comb * 255).astype(np.uint8)
            else:
                rgbf = colorize(tex)
                arr = (np.clip(rgbf, 0.0, 1.0) * 255).astype(np.uint8)
            new_img = Image.fromarray(arr)
            h, w = tex.shape
            canvas.config(width=w, height=h)
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)

    tk.Button(controls, text="Regenerate", command=do_regen).pack(side="left", padx=6)
    latlon_label = tk.Label(controls, textvariable=latlon_var, width=70, anchor="e")
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
    img = Image.fromarray(arr0)
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    LOG.info("Startup profile (top 20):")
    stats.print_stats(20)
    tk_img = ImageTk.PhotoImage(img)
    # Map area: canvas on the left, legend panel on the right (unused window space)
    map_row = tk.Frame(root)
    map_row.pack(side="top")
    canvas = tk.Canvas(map_row, width=tex0.shape[1], height=tex0.shape[0], highlightthickness=0)
    canvas.pack(side="left")
    img_id = canvas.create_image(0, 0, image=tk_img, anchor="nw")

    # --- Cursor tooltip: small floating panel that follows the mouse ---
    _tooltip_var = tk.StringVar(value="")
    _tooltip_win = tk.Toplevel(root)
    _tooltip_win.overrideredirect(True)
    _tooltip_win.withdraw()
    _tooltip_win.wm_attributes("-topmost", True)
    _tt_frame = tk.Frame(_tooltip_win, bg="#0d1117", bd=1, relief="solid")
    _tt_frame.pack()
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
        nonlocal tk_img, terrain_mode
        # Use simulation data if available and running
        use_sim_data = sim_state is not None and sim_running
        
        if mode_var.get() == "globe":
            with log_time("Render globe"):
                day = int(sim_state.day_of_year) if (use_sim_data and sim_state is not None) else 80
                view_name = view_var.get()
                if view_name == "Wind Arrows" or view_name == "Wind Particles":
                    view_name = "Wind"
                if view_name == "Ocean Temperature":
                    view_name = "Temperature"
                # In loaded mode, ensure elevation is cached before generating sphere
                if terrain_mode == "loaded":
                    elev_tex, _ = get_elevation_cache()
                    if elev_tex is None:
                        # Cache was cleared, shouldn't happen but fallback to procedural
                        terrain_mode = "procedural"
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
                nonlocal particle_anim_running
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
                if use_sim_data and sim_state.precipitation is not None:
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
                overlay, alpha = precipitation_to_rgb(P)
                comb = (1.0 - alpha[..., None]) * base_rgb + alpha[..., None] * overlay
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
                    C = sim_state.cloud_cover
                    # Grayscale cloud opacity: dark gray (thin) → white (dense)
                    C = np.clip(C.astype(np.float32), 0.0, 1.0)
                    gray = 0.25 + 0.75 * C
                    overlay = np.stack([gray, gray, gray], axis=-1)
                    alpha = C
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
                base_rgb = wind_speed_to_rgb(speed)
                arrows = render_wind_arrows(*tex.shape, u, v, target_arrows=int(wind_arrows_var.get()), scale=float(wind_scale_var.get()))
                arr = (np.clip(base_rgb + arrows, 0.0, 1.0) * 255).astype(np.uint8)
            elif view_var.get() == "Ocean Currents":
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
                speed = np.hypot(u, v).astype(np.float32)
                base_rgb = wind_speed_to_rgb(speed)
                arrows = render_wind_arrows(*tex.shape, u, v, target_arrows=int(wind_arrows_var.get()), scale=float(wind_scale_var.get()))
                ocean_mask, _ = get_masks(tex)
                mask3 = ocean_mask[..., None].astype(np.float32)
                comb = np.clip((base_rgb + arrows) * mask3, 0.0, 1.0)
                arr = (comb * 255).astype(np.uint8)
            else:
                arr = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            new_img = Image.fromarray(arr)
            h, w = tex.shape
            canvas.config(width=w, height=h)
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
            # On first Start: load autosave if enabled and available
            if _auto_save_enabled and AUTOSAVE_PATH.exists():
                try:
                    sim_state = load_state(AUTOSAVE_PATH)
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
        # If particle view is selected, start animation loop.
        nonlocal particle_anim_running
        if mode_var.get() == "map" and view_var.get() == "Wind Particles" and not particle_anim_running:
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
        render()
    
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
                if use_sim_data and sim_state.precipitation is not None:
                    precip = float(sim_state.precipitation[int(y), int(x)])
                else:
                    precip = 0.0
                if use_sim_data and sim_state.temperature is not None:
                    _T_disp = sim_state.air_temperature if sim_state.air_temperature is not None else sim_state.temperature
                    T_celsius = float(_T_disp[int(y), int(x)]) - 273.15
                else:
                    T_celsius = temperature_kelvin_for_lat(np.deg2rad(lat)) - 273.15
                latlon_var.set(f"lat {lat:6.2f}°, lon {lon:7.2f}°{px_str}, elev {alt_m:5.0f}m, T {T_celsius:.1f}°C")
                tt_lines = [hdr, f"Precip:  {precip:.3f} mm/day", f"Air T:   {T_celsius:.1f}°C"]

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
        # Autosave simulation state if enabled
        if _auto_save_enabled and sim_state is not None:
            try:
                save_state(sim_state, AUTOSAVE_PATH)
                LOG.info(f"Autosave written: day {sim_state.total_days:.0f}")
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
        }
        save_settings(s)
        graphs_controller.close()
        root.destroy()

    root.bind("<Escape>", lambda e: on_close())
    root.protocol("WM_DELETE_WINDOW", on_close)
    canvas.bind("<Motion>", on_motion)
    canvas.bind("<Leave>", lambda e: _tooltip_win.withdraw())
    root.mainloop()


if __name__ == "__main__":
    main()

