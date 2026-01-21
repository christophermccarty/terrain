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
from ocean import _ocean_mask_from_elevation, generate_ocean_currents
from terrain import precipitation_to_rgb
from simulate import PlanetState, create_initial_state, simulate_step, simulate_multiple_steps
from diagnostics import ClimateDiagnostics
import graphs

# Lightweight caches for expensive view layers
_WIND_CACHE = {"key": None, "u": None, "v": None}
_OCEAN_CURRENT_CACHE = {"key": None, "u": None, "v": None}
_PRECIP_VIEW_CACHE = {"key": None, "P": None}


class SimulationThread(Thread):
    """Background thread that runs physics simulation independently of UI."""

    def __init__(self, initial_state, days_per_step=1.0, wind_block_size=8, diagnostics=None):
        super().__init__(daemon=True)
        self.state = initial_state
        self.days_per_step = days_per_step
        self.wind_block_size = wind_block_size
        self.diagnostics = diagnostics
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
                # Run one simulation step
                new_state, temp_components = simulate_step(
                    self.state,
                    days=self.days_per_step,
                    wind_block_size=self.wind_block_size,
                    debug_log=False,
                    track_components=True,
                )
                self.state = new_state

                # Record diagnostics if available
                if self.diagnostics is not None:
                    self.diagnostics.record_step(
                        new_state,
                        new_state.day_of_year,
                        days_elapsed=self.days_per_step,
                        component_contributions=temp_components
                    )

                # Push to UI (non-blocking, drop if UI busy)
                try:
                    self.state_queue.put_nowait(new_state)
                    self.component_queue.put_nowait(temp_components)
                except:
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
        """Update simulation speed."""
        self.days_per_step = days

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
        # Run for 365 days
        states, _ = simulate_multiple_steps(sim_state, total_days=365.0, step_days=1.0)
        
        # Analyze final state
        final_state = states[-1]
        stats = diagnostics.analyze_snapshot(final_state)
        diagnostics.print_report(stats)
        # Circulation validation (surface 3-cell proxies)
        circ = diagnostics.analyze_circulation(stats)
        diagnostics.print_circulation_report(circ)
        
        messagebox.showinfo("Benchmark Complete", 
            f"Global Mean Temp: {stats['global_mean_temp']:.1f} K\n"
            f"Equator-Pole Gradient: {stats['gradient_north']:.1f} K\n"
            "Check console for full report.")
        
        sim_status_var.set("Stopped")

    tk.Button(controls, text="Export Data", command=export_data).pack(side="right", padx=10)
    tk.Button(controls, text="Benchmark", command=run_benchmark).pack(side="right", padx=10)

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
        sim_cycle_var.set("Year: 1")
        graphs_controller.reset()
    
    # Simulation controls
    sim_controls = tk.Frame(root)
    sim_controls.pack(fill="x")
    sim_status_var = tk.StringVar(value="Stopped")
    sim_cycle_var = tk.StringVar(value="Year: 1")
    tk.Label(sim_controls, text="Simulation:").pack(side="left", padx=(4,0))
    sim_status_label = tk.Label(sim_controls, textvariable=sim_status_var)
    sim_status_label.pack(side="left", padx=4)
    sim_cycle_label = tk.Label(sim_controls, textvariable=sim_cycle_var)
    sim_cycle_label.pack(side="left", padx=8)
    tk.Button(sim_controls, text="Start", command=lambda: start_simulation()).pack(side="left", padx=2)
    tk.Button(sim_controls, text="Stop", command=lambda: stop_simulation()).pack(side="left", padx=2)
    tk.Button(sim_controls, text="Pause", command=lambda: pause_simulation()).pack(side="left", padx=2)
    tk.Button(sim_controls, text="Reset", command=lambda: reset_simulation()).pack(side="left", padx=2)
    def on_graphs_toggle():
        graphs_controller.set_enabled(graphs_enabled_var.get())
    tk.Checkbutton(sim_controls, text="Graphs", variable=graphs_enabled_var, command=on_graphs_toggle).pack(side="left", padx=6)

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
                    u, v = generate_ocean_currents(tex, day_of_year=day, time_days=float(day))
                    _OCEAN_CURRENT_CACHE.update({"key": ckey, "u": u, "v": v})
                else:
                    u, v = _OCEAN_CURRENT_CACHE["u"], _OCEAN_CURRENT_CACHE["v"]
                speed = np.hypot(u, v).astype(np.float32)
                base_rgb = wind_speed_to_rgb(speed)
                arrows = render_wind_arrows(*tex.shape, u, v, target_arrows=int(wind_arrows_var.get()), scale=float(wind_scale_var.get()))
                ocean_mask = _ocean_mask_from_elevation(tex)
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
    latlon_label = tk.Label(controls, textvariable=latlon_var)
    latlon_label.pack(side="right")

    # Profile initial generation once at startup
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Try to load default heightmap, fall back to procedural if not found
    default_heightmap_path = r"D:\dev\planetsim\images\16_bit_dem_small_512.tif"
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
    canvas = tk.Canvas(root, width=tex0.shape[1], height=tex0.shape[0], highlightthickness=0)
    canvas.pack()
    img_id = canvas.create_image(0, 0, image=tk_img, anchor="nw")

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
                    # Use simulation temperature, convert to overlay RGB using same high-quality mapping
                    from temperature import temperature_to_rgb
                    T = sim_state.temperature
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
                ocean_mask = _ocean_mask_from_elevation(tex)
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
                ocean_mask = _ocean_mask_from_elevation(tex)
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
            sim_status_var.set(f"Day: {sim_state.day_of_year:.1f}")
        # Clear lat/lon when switching out of map
        if mode_var.get() != "map":
            latlon_var.set("")

    def start_simulation():
        nonlocal sim_state, sim_thread, sim_running, sim_paused
        if sim_state is None:
            # Initialize simulation from current elevation
            tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
            _init_sim_state_from_elevation(tex)

        # Start or resume simulation thread
        if sim_thread is None or not sim_thread.is_alive():
            sim_thread = SimulationThread(
                sim_state,
                days_per_step=sim_speed,
                wind_block_size=int(wind_block_size_var.get()),
                diagnostics=diagnostics
            )
            sim_thread.start()

        sim_thread.resume()
        sim_running = True
        sim_paused = False
        sim_status_var.set("Running")
        # Diagnostics tracks total time; use it to display cycle count (years).
        sim_cycle_var.set(f"Year: {int(diagnostics.total_days // 365.2422) + 1}")
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
        """Update the cursor display for given canvas coordinates."""
        if mode_var.get() != "map":
            return
        use_sim_data = sim_state is not None and sim_running
        if use_sim_data and sim_state.elevation is not None:
            tex = sim_state.elevation
        else:
            tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
        h, w = tex.shape
        if 0 <= x < w and 0 <= y < h:
            # Equirectangular: x∈[0,w) -> φ∈[-π,π), y∈[0,h) -> θ∈[-π/2,π/2]
            lon = (x / w) * 360.0 - 180.0
            lat = 90.0 - (y / h) * 180.0
            # Calculate elevation/altitude
            elev_raw = float(tex[int(y), int(x)])
            # Check if using loaded heightmap (generic normalization) or procedural (power-law)
            if terrain_mode == "loaded":
                # Generic elevation mapping (works with any heightmap):
                # 0.0: Ocean = 0m
                # 0.0-0.03: Low-lying (rivers, basins, coastal) = 0-100m linear
                # 0.03-1.0: Higher elevations = 100-8848m (power curve x^2.5)
                pixel_display = f"norm {elev_raw:.3f}"
                if elev_raw == 0.0:
                    alt_m = 0.0  # Ocean
                elif elev_raw <= 0.03:
                    # Low-lying areas: 0-100m linear
                    alt_m = (elev_raw / 0.03) * 100.0
                else:
                    # Higher elevations: 100m to 8848m using power curve
                    normalized = (elev_raw - 0.03) / 0.97  # 0.0 to 1.0
                    alt_m = 100.0 + (normalized ** 2.5) * 8748.0
            else:
                # Procedural terrain uses power-law with sea level at 0.2
                pixel_display = ""  # No pixel value for procedural
                sea_level = 0.2
                if elev_raw <= sea_level:
                    alt_m = 0.0
                else:
                    elevation_above_sea = elev_raw - sea_level
                    alt_m = (elevation_above_sea / (1.0 - sea_level)) ** 2.0 * 8848.0
            if view_var.get() == "Wind Arrows" or view_var.get() == "Wind Particles":
                if use_sim_data and sim_state.wind_u is not None and sim_state.wind_v is not None:
                    u, v = sim_state.wind_u, sim_state.wind_v
                elif _WIND_CACHE["u"] is None or _WIND_CACHE["u"].shape != (h, w):
                    u, v = generate_wind_field(h, w, elevation=tex, upsample="bilinear", weather_amp=0.35, debug_log=False)
                    _WIND_CACHE.update({"key": (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()), "bilinear"), "u": u, "v": v})
                else:
                    u, v = _WIND_CACHE["u"], _WIND_CACHE["v"]
                speed = float(np.hypot(u[int(y), int(x)], v[int(y), int(x)]))
                px_str = f", {pixel_display}" if pixel_display else ""
                latlon_var.set(f"lat {lat:.2f}°, lon {lon:.2f}°{px_str}, elev {alt_m:.0f}m, wind {speed:.1f} m/s")
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
                speed = float(np.hypot(u[int(y), int(x)], v[int(y), int(x)]))
                px_str = f", {pixel_display}" if pixel_display else ""
                latlon_var.set(f"lat {lat:.2f}°, lon {lon:.2f}°{px_str}, elev {alt_m:.0f}m, current {speed:.2f} m/s")
            else:
                # Use simulation temperature if available
                if use_sim_data and sim_state.temperature is not None:
                    T_kelvin = float(sim_state.temperature[int(y), int(x)])
                else:
                    T_kelvin = temperature_kelvin_for_lat(np.deg2rad(lat))
                # Convert Kelvin to Celsius
                T_celsius = T_kelvin - 273.15
                px_str = f", {pixel_display}" if pixel_display else ""
                latlon_var.set(f"lat {lat:.2f}°, lon {lon:.2f}°{px_str}, elev {alt_m:.0f}m, T {T_celsius:.1f}°C")
        else:
            latlon_var.set("")
    
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
                sim_cycle_var.set(f"Year: {int(diagnostics.total_days // 365.2422) + 1}")
                if not sim_paused:
                    sim_status_var.set(f"Day: {sim_state.day_of_year:.1f}")

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
        nonlocal sim_thread
        # Stop simulation thread if running
        if sim_thread:
            sim_thread.stop()
            sim_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
        # Persist current UI parameters to settings.json
        s = {
            "seed": int(seed_var.get()),
            "octaves": int(octaves_var.get()),
            "freq": float(freq_var.get()),
            "lac": float(lac_var.get()),
            "gain": float(gain_var.get()),
            "wind_arrows": int(wind_arrows_var.get()),
            "wind_scale": float(wind_scale_var.get()),
            "wind_block_size": int(wind_block_size_var.get()),
        }
        save_settings(s)
        graphs_controller.close()
        root.destroy()

    root.bind("<Escape>", lambda e: on_close())
    root.protocol("WM_DELETE_WINDOW", on_close)
    canvas.bind("<Motion>", on_motion)
    root.mainloop()


if __name__ == "__main__":
    main()

