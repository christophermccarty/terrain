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
from atmosphere import generate_wind_field, render_wind_arrows, generate_precipitation
from temperature import generate_temperature_overlay, temperature_kelvin_for_lat
from simulate import PlanetState, create_initial_state, simulate_step

# Lightweight caches for expensive view layers
_WIND_CACHE = {"key": None, "u": None, "v": None}
_PRECIP_CACHE = {"key": None, "P": None}


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
    }

    root = tk.Tk()
    root.title(f"Sphere {size}x{size} (262,144 cells)")
    root.resizable(False, False)

    # Simulation state
    sim_state: PlanetState | None = None
    sim_running = False
    sim_paused = False
    sim_speed = 1.0  # days per step
    last_mouse_pos = (0, 0)  # Track last mouse position for cursor updates
    last_debug_day = 0.0  # Track last day we logged debug info
    
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
    tk.OptionMenu(controls, view_var, "Terrain", "Temperature", "Wind", "Precipitation").pack(side="left")
    
    # Simulation controls
    sim_controls = tk.Frame(root)
    sim_controls.pack(fill="x")
    sim_status_var = tk.StringVar(value="Stopped")
    tk.Label(sim_controls, text="Simulation:").pack(side="left", padx=(4,0))
    sim_status_label = tk.Label(sim_controls, textvariable=sim_status_var)
    sim_status_label.pack(side="left", padx=4)
    tk.Button(sim_controls, text="Start", command=lambda: start_simulation()).pack(side="left", padx=2)
    tk.Button(sim_controls, text="Stop", command=lambda: stop_simulation()).pack(side="left", padx=2)
    tk.Button(sim_controls, text="Pause", command=lambda: pause_simulation()).pack(side="left", padx=2)
    tk.Button(sim_controls, text="Reset", command=lambda: reset_simulation()).pack(side="left", padx=2)

    # Wind controls
    wind_arrows_var = tk.IntVar(value=int(settings.get("wind_arrows", default_settings["wind_arrows"])))
    wind_scale_var = tk.DoubleVar(value=float(settings.get("wind_scale", default_settings["wind_scale"])))
    precip_evap_var = tk.DoubleVar(value=float(settings.get("precip_evap_coeff", 1.0)))
    precip_uplift_var = tk.DoubleVar(value=float(settings.get("precip_uplift_coeff", 1.0)))
    precip_eff_var = tk.DoubleVar(value=float(settings.get("precip_efficiency", 0.7)))
    def add_wind_controls():
        frm = tk.Frame(root)
        frm.pack(fill="x")
        def add(parent, label, var, width=6):
            f = tk.Frame(parent); f.pack(side="left", padx=4); tk.Label(f, text=label).pack(side="left"); tk.Entry(f, textvariable=var, width=width).pack(side="left")
        add(frm, "Arrows", wind_arrows_var)
        add(frm, "Scale", wind_scale_var)
        add(frm, "Evap", precip_evap_var)
        add(frm, "Uplift", precip_uplift_var)
        add(frm, "Eff", precip_eff_var)
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
                
                # Reset simulation
                sim_state = None
                sim_running = False
                sim_paused = False
                sim_status_var.set("Stopped")
                
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
        
        # Reset simulation
        sim_state = None
        sim_running = False
        sim_paused = False
        sim_status_var.set("Stopped")
        
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
            new_img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll), view=view_var.get(), day_of_year=day, **p)
            canvas.config(width=size, height=size)
        else:
            tex = ensure_elevation(size, **p)
            if view_var.get() == "Wind":
                base_rgb = colorize(tex)
                wkey = (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()))
                if _WIND_CACHE["key"] != wkey:
                    with log_time("Generate wind field+arrows"):
                        u, v = generate_wind_field(*tex.shape, elevation=tex, debug_log=False)
                        _WIND_CACHE.update({"key": wkey, "u": u, "v": v})
                else:
                    u, v = _WIND_CACHE["u"], _WIND_CACHE["v"]
                arrows = render_wind_arrows(*tex.shape, u, v, target_arrows=int(wind_arrows_var.get()), scale=float(wind_scale_var.get()))
                comb = np.clip(base_rgb + arrows, 0.0, 1.0)
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
                # In loaded mode, ensure elevation is cached before generating sphere
                if terrain_mode == "loaded":
                    elev_tex, _ = get_elevation_cache()
                    if elev_tex is None:
                        # Cache was cleared, shouldn't happen but fallback to procedural
                        terrain_mode = "procedural"
                new_img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll), view=view_var.get(), seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get(), day_of_year=day)
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
            base_rgb = colorize(tex)
            
            if view_var.get() == "Temperature":
                if use_sim_data and sim_state.temperature is not None:
                    # Use simulation temperature, convert to overlay RGB
                    T = sim_state.temperature
                    h, w = T.shape
                    T_norm = (T - 200.0) / 100.0  # Normalize roughly [200K, 300K] -> [0, 1]
                    T_norm = np.clip(T_norm, 0.0, 1.0)
                    # Blue→Green→Yellow→Red gradient
                    cstops = np.array([[0.0,0.2,0.8],[0.0,0.8,0.0],[0.9,0.9,0.0],[0.9,0.1,0.0]], dtype=np.float32)
                    bp = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
                    i = np.clip(np.searchsorted(bp, T_norm, side="right") - 1, 0, len(bp) - 2)
                    c0 = cstops[i]; c1 = cstops[i+1]
                    t = (T_norm - bp[i]) / (bp[i+1] - bp[i] + 1e-9)
                    overlay = c0 + (c1 - c0) * t[..., None]
                else:
                    h, w = tex.shape
                    with log_time("Generate temperature overlay"):
                        overlay = generate_temperature_overlay(h, w, elevation=tex)
                alpha = 0.5
                comb = (1.0 - alpha) * base_rgb + alpha * overlay
                arr = (np.clip(comb, 0.0, 1.0) * 255).astype(np.uint8)
            elif view_var.get() == "Wind":
                if use_sim_data and sim_state.wind_u is not None and sim_state.wind_v is not None:
                    u, v = sim_state.wind_u, sim_state.wind_v
                else:
                    wkey = (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()))
                    if _WIND_CACHE["key"] != wkey:
                        with log_time("Generate wind field"):
                            u, v = generate_wind_field(*tex.shape, elevation=tex, debug_log=False)
                            _WIND_CACHE.update({"key": wkey, "u": u, "v": v})
                    else:
                        u, v = _WIND_CACHE["u"], _WIND_CACHE["v"]
                arrows = render_wind_arrows(*tex.shape, u, v, target_arrows=int(wind_arrows_var.get()), scale=float(wind_scale_var.get()))
                arr = (np.clip(base_rgb + arrows, 0.0, 1.0) * 255).astype(np.uint8)
            elif view_var.get() == "Precipitation":
                if use_sim_data and sim_state.precipitation is not None:
                    P = sim_state.precipitation
                else:
                    pkey = (tex.shape, float(precip_evap_var.get()), float(precip_uplift_var.get()), float(precip_eff_var.get()))
                    if _PRECIP_CACHE["key"] != pkey:
                        with log_time("Generate precipitation"):
                            day = int(sim_state.day_of_year) if (use_sim_data and sim_state is not None) else 80
                            P, _, _ = generate_precipitation(
                                *tex.shape,
                                tex,
                                day_of_year=day,
                                evap_coeff=float(precip_evap_var.get()),
                                uplift_coeff=float(precip_uplift_var.get()),
                                rain_efficiency=float(precip_eff_var.get()),
                            )
                        _PRECIP_CACHE.update({"key": pkey, "P": P})
                    else:
                        P = _PRECIP_CACHE["P"]
                # Fixed color scale: 0..20 mm/day
                v = np.clip(P / 20.0, 0.0, 1.0).astype(np.float32)
                LOG.info(f"Precip stats: min {float(np.min(P)):.2f}, mean {float(np.mean(P)):.2f}, max {float(np.max(P)):.2f} mm/day")
                # Blue→Green→Yellow→Red
                cstops = np.array([[0.0,0.2,0.8],[0.0,0.8,0.0],[0.9,0.9,0.0],[0.9,0.1,0.0]], dtype=np.float32)
                bp = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
                i = np.clip(np.searchsorted(bp, v, side="right") - 1, 0, len(bp) - 2)
                c0 = cstops[i]; c1 = cstops[i+1]
                t = (v - bp[i]) / (bp[i+1] - bp[i] + 1e-9)
                rgb = c0 + (c1 - c0) * t[..., None]
                arr = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                arr = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            new_img = Image.fromarray(arr)
            h, w = tex.shape
            canvas.config(width=w, height=h)
        tk_img = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(img_id, image=tk_img)
        # Update status with simulation day (only if not paused)
        if use_sim_data and not sim_paused:
            sim_status_var.set(f"Running: Day {sim_state.day_of_year:.1f}")
        # Clear lat/lon when switching out of map
        if mode_var.get() != "map":
            latlon_var.set("")

    def start_simulation():
        nonlocal sim_state, sim_running, sim_paused, last_debug_day
        if sim_state is None:
            # Initialize simulation from current elevation
            tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
            sim_state = create_initial_state(
                tex,
                day_of_year=1.0,
                evap_coeff=float(precip_evap_var.get()),
                uplift_coeff=float(precip_uplift_var.get()),
                rain_efficiency=float(precip_eff_var.get()),
            )
            last_debug_day = sim_state.day_of_year  # Initialize debug tracking
        sim_running = True
        sim_paused = False
        sim_status_var.set("Running")
        update_simulation()
    
    def stop_simulation():
        nonlocal sim_running, sim_paused
        sim_running = False
        sim_paused = False
        sim_status_var.set("Stopped")
    
    def pause_simulation():
        nonlocal sim_paused
        if sim_running:
            sim_paused = not sim_paused
            sim_status_var.set("Paused" if sim_paused else "Running")
    
    def reset_simulation():
        nonlocal sim_state, sim_running, sim_paused
        sim_state = None
        sim_running = False
        sim_paused = False
        sim_status_var.set("Stopped")
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
            if view_var.get() == "Wind":
                if use_sim_data and sim_state.wind_u is not None and sim_state.wind_v is not None:
                    u, v = sim_state.wind_u, sim_state.wind_v
                elif _WIND_CACHE["u"] is None or _WIND_CACHE["u"].shape != (h, w):
                    u, v = generate_wind_field(h, w, elevation=tex, debug_log=False)
                    _WIND_CACHE.update({"key": (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get())), "u": u, "v": v})
                else:
                    u, v = _WIND_CACHE["u"], _WIND_CACHE["v"]
                speed = float(np.hypot(u[int(y), int(x)], v[int(y), int(x)]))
                px_str = f", {pixel_display}" if pixel_display else ""
                latlon_var.set(f"lat {lat:.2f}°, lon {lon:.2f}°{px_str}, elev {alt_m:.0f}m, wind {speed:.1f} m/s")
            elif view_var.get() == "Precipitation":
                if use_sim_data and sim_state.precipitation is not None:
                    P = sim_state.precipitation
                elif _PRECIP_CACHE["P"] is None or _PRECIP_CACHE["P"].shape != (h, w):
                    day = int(sim_state.day_of_year) if (use_sim_data and sim_state is not None) else 80
                    P, _, _ = generate_precipitation(
                        h,
                        w,
                        tex,
                        day_of_year=day,
                        evap_coeff=float(precip_evap_var.get()),
                        uplift_coeff=float(precip_uplift_var.get()),
                        rain_efficiency=float(precip_eff_var.get()),
                    )
                    _PRECIP_CACHE.update({"key": (tex.shape, float(precip_evap_var.get()), float(precip_uplift_var.get()), float(precip_eff_var.get())), "P": P})
                else:
                    P = _PRECIP_CACHE["P"]
                px_str = f", {pixel_display}" if pixel_display else ""
                latlon_var.set(f"lat {lat:.2f}°, lon {lon:.2f}°{px_str}, elev {alt_m:.0f}m, precip {float(P[int(y), int(x)]):.2f} mm/day")
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
    
    def update_simulation():
        nonlocal sim_state, last_debug_day
        if sim_running and not sim_paused and sim_state is not None:
            # Check if we should log debug info (every 10 days)
            should_log = (sim_state.day_of_year - last_debug_day) >= 10.0
            if should_log:
                last_debug_day = sim_state.day_of_year
            
            # Advance simulation
            sim_state = simulate_step(
                sim_state,
                days=sim_speed,
                evap_coeff=float(precip_evap_var.get()),
                uplift_coeff=float(precip_uplift_var.get()),
                rain_efficiency=float(precip_eff_var.get()),
                debug_log=should_log,
            )
            # Update display
            render()
            # Update cursor display at last known mouse position
            if mode_var.get() == "map":
                update_cursor_display(last_mouse_pos[0], last_mouse_pos[1])
            # Schedule next update (100ms = ~10 FPS)
            root.after(100, update_simulation)
    
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
        # Persist current UI parameters to settings.json
        s = {
            "seed": int(seed_var.get()),
            "octaves": int(octaves_var.get()),
            "freq": float(freq_var.get()),
            "lac": float(lac_var.get()),
            "gain": float(gain_var.get()),
            "wind_arrows": int(wind_arrows_var.get()),
            "wind_scale": float(wind_scale_var.get()),
            "precip_evap_coeff": float(precip_evap_var.get()),
            "precip_uplift_coeff": float(precip_uplift_var.get()),
            "precip_efficiency": float(precip_eff_var.get()),
        }
        save_settings(s)
        root.destroy()

    root.bind("<Escape>", lambda e: on_close())
    root.protocol("WM_DELETE_WINDOW", on_close)
    canvas.bind("<Motion>", on_motion)
    root.mainloop()


if __name__ == "__main__":
    main()

