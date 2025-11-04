"""Main entry point for planet simulator.

Launches the GUI application for viewing and interacting with the planet simulation.
All modules are kept separated: terrain, atmosphere, temperature, and simulate.
"""

import tkinter as tk
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
            new_img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll), view=view_var.get(), **p)
            canvas.config(width=size, height=size)
        else:
            tex = ensure_elevation(size, **p)
            if view_var.get() == "Wind":
                base_rgb = colorize(tex)
                wkey = (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()))
                if _WIND_CACHE["key"] != wkey:
                    with log_time("Generate wind field+arrows"):
                        u, v = generate_wind_field(*tex.shape)
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
    # Default: Map view
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
        nonlocal tk_img
        # Use simulation data if available and running
        use_sim_data = sim_state is not None and sim_running
        
        if mode_var.get() == "globe":
            with log_time("Render globe"):
                new_img = generate_sphere_image(size=size, radius=0.96, rot=(yaw, pitch, roll), view=view_var.get(), seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
            canvas.config(width=size, height=size)
        else:
            # Get elevation from simulation or generate
            if use_sim_data and sim_state.elevation is not None:
                tex = sim_state.elevation
            else:
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
                        overlay = generate_temperature_overlay(h, w)
                alpha = 0.7
                comb = (1.0 - alpha) * base_rgb + alpha * overlay
                arr = (np.clip(comb, 0.0, 1.0) * 255).astype(np.uint8)
            elif view_var.get() == "Wind":
                if use_sim_data and sim_state.wind_u is not None and sim_state.wind_v is not None:
                    u, v = sim_state.wind_u, sim_state.wind_v
                else:
                    wkey = (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get()))
                    if _WIND_CACHE["key"] != wkey:
                        with log_time("Generate wind field"):
                            u, v = generate_wind_field(*tex.shape)
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
                            P, _ = generate_precipitation(*tex.shape, tex, evap_coeff=float(precip_evap_var.get()), uplift_coeff=float(precip_uplift_var.get()), rain_efficiency=float(precip_eff_var.get()))
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
        nonlocal sim_state, sim_running, sim_paused
        if sim_state is None:
            # Initialize simulation from current elevation
            tex = ensure_elevation(size, seed=seed_var.get(), octaves=octaves_var.get(), freq=freq_var.get(), lac=lac_var.get(), gain=gain_var.get())
            sim_state = create_initial_state(
                tex,
                day_of_year=80.0,
                evap_coeff=float(precip_evap_var.get()),
                uplift_coeff=float(precip_uplift_var.get()),
                rain_efficiency=float(precip_eff_var.get()),
            )
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
            if view_var.get() == "Wind":
                if use_sim_data and sim_state.wind_u is not None and sim_state.wind_v is not None:
                    u, v = sim_state.wind_u, sim_state.wind_v
                elif _WIND_CACHE["u"] is None or _WIND_CACHE["u"].shape != (h, w):
                    u, v = generate_wind_field(h, w)
                    _WIND_CACHE.update({"key": (tex.shape, int(wind_arrows_var.get()), float(wind_scale_var.get())), "u": u, "v": v})
                else:
                    u, v = _WIND_CACHE["u"], _WIND_CACHE["v"]
                speed = float(np.hypot(u[int(y), int(x)], v[int(y), int(x)]))
                latlon_var.set(f"lat {lat:.2f}°, lon {lon:.2f}°, wind {speed:.1f} m/s")
            elif view_var.get() == "Precipitation":
                if use_sim_data and sim_state.precipitation is not None:
                    P = sim_state.precipitation
                elif _PRECIP_CACHE["P"] is None or _PRECIP_CACHE["P"].shape != (h, w):
                    P, _ = generate_precipitation(h, w, tex)
                    _PRECIP_CACHE.update({"key": (tex.shape, float(precip_evap_var.get()), float(precip_uplift_var.get()), float(precip_eff_var.get())), "P": P})
                else:
                    P = _PRECIP_CACHE["P"]
                latlon_var.set(f"lat {lat:.2f}°, lon {lon:.2f}°, precip {float(P[int(y), int(x)]):.2f} mm/day")
            else:
                # Use simulation temperature if available
                if use_sim_data and sim_state.temperature is not None:
                    T = float(sim_state.temperature[int(y), int(x)])
                else:
                    T = temperature_kelvin_for_lat(np.deg2rad(lat))
                latlon_var.set(f"lat {lat:.2f}°, lon {lon:.2f}°, T {T:.1f} K")
        else:
            latlon_var.set("")
    
    def update_simulation():
        nonlocal sim_state
        if sim_running and not sim_paused and sim_state is not None:
            # Advance simulation
            sim_state = simulate_step(
                sim_state,
                days=sim_speed,
                evap_coeff=float(precip_evap_var.get()),
                uplift_coeff=float(precip_uplift_var.get()),
                rain_efficiency=float(precip_eff_var.get()),
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

