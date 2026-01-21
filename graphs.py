"""Live matplotlib graphs for simulation diagnostics."""

from __future__ import annotations

from collections import deque
import time
from typing import Callable, Optional

import numpy as np
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ocean import _ocean_mask_from_elevation


class LiveGraphsWindow:
    """Tk window that hosts live matplotlib subplots."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        history_days: float = 365.0,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self.history_days = float(history_days)
        self._on_close_cb = on_close
        self._last_total_days: Optional[float] = None

        self.top = tk.Toplevel(root)
        self.top.title("Simulation Graphs")
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fig = Figure(figsize=(20, 9), dpi=100)
        axes = self.fig.subplots(4, 4)
        self.axes = axes

        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)

        self._times: deque[float] = deque()
        self._series: dict[str, deque[float]] = {
            "avg_temp": deque(),
            "avg_ocean_temp": deque(),
            "wind_mean": deque(),
            "wind_min": deque(),
            "wind_max": deque(),
            "net_radiation": deque(),
            "temp_std": deque(),
            "circulation_score": deque(),
            "ice_cover_fraction": deque(),
            "cloud_mean": deque(),
            "albedo_mean": deque(),
            "wind_div_rms": deque(),
            "temp_min": deque(),
            "temp_max": deque(),
            "temp_equator": deque(),
            "temp_pole_north": deque(),
            "temp_pole_south": deque(),
            "wind_trade_mean": deque(),
            "wind_midlat_mean": deque(),
            "precip_mean": deque(),
            "precip_min": deque(),
            "precip_max": deque(),
            "precip_equator": deque(),
            "precip_pole_north": deque(),
            "precip_pole_south": deque(),
            "humidity_mean": deque(),
            "humidity_min": deque(),
            "humidity_max": deque(),
            "soil_mean": deque(),
            "soil_min": deque(),
            "soil_max": deque(),
        }

        self._lines: dict[str, any] = {}
        self._init_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.fig.tight_layout(pad=2.0)
        self._position_right_of_root(root)

    def _init_axes(self) -> None:
        axes = list(self.axes.ravel())
        ax00 = axes[0]
        ax01 = axes[1]
        ax02 = axes[2]
        ax03 = axes[3]
        ax10 = axes[4]
        ax11 = axes[5]
        ax12 = axes[6]
        ax13 = axes[7]
        ax20 = axes[8]
        ax21 = axes[9]
        ax22 = axes[10]
        ax23 = axes[11]
        ax30 = axes[12]
        ax31 = axes[13]
        ax32 = axes[14]
        ax33 = axes[15]

        ax00.set_title("Avg Temperature (K)")
        self._lines["avg_temp"] = ax00.plot([], [], color="tab:red")[0]

        ax01.set_title("Avg Ocean Temperature (K)")
        self._lines["avg_ocean_temp"] = ax01.plot([], [], color="tab:blue")[0]

        ax02.set_title("Wind Speed (m/s)")
        self._lines["wind_mean"] = ax02.plot([], [], color="tab:green", label="Mean")[0]
        self._lines["wind_min"] = ax02.plot([], [], color="tab:gray", label="Min")[0]
        self._lines["wind_max"] = ax02.plot([], [], color="tab:orange", label="Max")[0]
        ax02.legend(loc="upper right", fontsize=8)

        ax03.set_title("Net Radiation (W/m^2)")
        self._lines["net_radiation"] = ax03.plot([], [], color="tab:purple")[0]

        ax10.set_title("Temperature Std Dev (K)")
        self._lines["temp_std"] = ax10.plot([], [], color="tab:brown")[0]

        ax11.set_title("Circulation Score")
        self._lines["circulation_score"] = ax11.plot([], [], color="tab:olive")[0]

        ax12.set_title("Ice Cover Fraction")
        self._lines["ice_cover_fraction"] = ax12.plot([], [], color="tab:cyan")[0]

        ax13.set_title("Cloud/Albedo Mean")
        self._lines["cloud_mean"] = ax13.plot([], [], color="tab:blue", label="Cloud")[0]
        self._lines["albedo_mean"] = ax13.plot([], [], color="tab:pink", label="Albedo")[0]
        ax13.legend(loc="upper right", fontsize=8)

        ax20.set_title("Wind Divergence RMS")
        self._lines["wind_div_rms"] = ax20.plot([], [], color="tab:brown")[0]

        ax21.set_title("Temperature Min/Max (K)")
        self._lines["temp_min"] = ax21.plot([], [], color="tab:gray", label="Min")[0]
        self._lines["temp_max"] = ax21.plot([], [], color="tab:red", label="Max")[0]
        ax21.legend(loc="upper right", fontsize=8)

        ax22.set_title("Equator/Pole Temps (K)")
        self._lines["temp_equator"] = ax22.plot([], [], color="tab:orange", label="Equator")[0]
        self._lines["temp_pole_north"] = ax22.plot([], [], color="tab:blue", label="Pole N")[0]
        self._lines["temp_pole_south"] = ax22.plot([], [], color="tab:cyan", label="Pole S")[0]
        ax22.legend(loc="upper right", fontsize=8)

        ax23.set_title("Wind Bands (m/s)")
        self._lines["wind_trade_mean"] = ax23.plot([], [], color="tab:green", label="Trades")[0]
        self._lines["wind_midlat_mean"] = ax23.plot([], [], color="tab:olive", label="Mid-lat")[0]
        ax23.legend(loc="upper right", fontsize=8)

        ax30.set_title("Precipitation Mean/Range (mm/day)")
        self._lines["precip_mean"] = ax30.plot([], [], color="tab:blue", label="Mean")[0]
        self._lines["precip_min"] = ax30.plot([], [], color="tab:gray", label="Min")[0]
        self._lines["precip_max"] = ax30.plot([], [], color="tab:purple", label="Max")[0]
        ax30.legend(loc="upper right", fontsize=8)

        ax31.set_title("Precipitation by Latitude (mm/day)")
        self._lines["precip_equator"] = ax31.plot([], [], color="tab:red", label="Equator")[0]
        self._lines["precip_pole_north"] = ax31.plot([], [], color="tab:blue", label="Pole N")[0]
        self._lines["precip_pole_south"] = ax31.plot([], [], color="tab:cyan", label="Pole S")[0]
        ax31.legend(loc="upper right", fontsize=8)

        ax32.set_title("Humidity Mean/Range (kg/kg)")
        self._lines["humidity_mean"] = ax32.plot([], [], color="tab:blue", label="Mean")[0]
        self._lines["humidity_min"] = ax32.plot([], [], color="tab:gray", label="Min")[0]
        self._lines["humidity_max"] = ax32.plot([], [], color="tab:purple", label="Max")[0]
        ax32.legend(loc="upper right", fontsize=8)

        ax33.set_title("Soil Moisture Mean/Range")
        self._lines["soil_mean"] = ax33.plot([], [], color="tab:green", label="Mean")[0]
        self._lines["soil_min"] = ax33.plot([], [], color="tab:gray", label="Min")[0]
        self._lines["soil_max"] = ax33.plot([], [], color="tab:olive", label="Max")[0]
        ax33.legend(loc="upper right", fontsize=8)

    def reset(self) -> None:
        self._times.clear()
        for series in self._series.values():
            series.clear()
        self._last_total_days = None
        for key, line in self._lines.items():
            line.set_data([], [])
        for ax in self.axes.ravel():
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()

    def _position_right_of_root(self, root: tk.Tk) -> None:
        root.update_idletasks()
        self.top.update_idletasks()
        root_x = root.winfo_x()
        root_y = root.winfo_y()
        root_w = root.winfo_width()
        win_w = self.top.winfo_reqwidth()
        win_h = self.top.winfo_reqheight()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = min(root_x + root_w + 10, screen_w - win_w)
        x = max(0, x)
        y = min(max(0, root_y), max(0, screen_h - win_h))
        self.top.geometry(f"{win_w}x{win_h}+{x}+{y}")

    def _on_close(self) -> None:
        if self._on_close_cb is not None:
            self._on_close_cb()
        self.top.destroy()

    def close(self) -> None:
        self.top.destroy()

    def update_from(self, state, diagnostics) -> None:
        if state is None or state.temperature is None:
            return
        stats = diagnostics.history[-1] if diagnostics.history else {}
        total_days = float(stats.get("total_days", diagnostics.total_days))
        if self._last_total_days is not None and total_days == self._last_total_days:
            return
        self._last_total_days = total_days

        self._times.append(total_days)

        avg_temp = float(stats.get("global_mean_temp", np.mean(state.temperature)))
        avg_ocean_temp = self._ocean_mean_temp(state)
        wind_mean, wind_min, wind_max = self._wind_stats(state)
        net_radiation = self._net_radiation(diagnostics)
        temp_std = float(stats.get("T_std", np.std(state.temperature)))
        circulation_score = float(stats.get("circulation_score", 0.0))
        ice_cover_fraction = float(stats.get("ice_cover_fraction", 0.0))
        cloud_mean = float(stats.get("cloud_mean", 0.0))
        albedo_mean = float(stats.get("albedo_mean", 0.3))
        wind_div_rms = float(stats.get("wind_div_rms", 0.0))
        temp_min = float(stats.get("T_min", np.min(state.temperature)))
        temp_max = float(stats.get("T_max", np.max(state.temperature)))
        temp_equator = float(stats.get("T_equator", 0.0))
        temp_pole_north = float(stats.get("T_pole_north", 0.0))
        temp_pole_south = float(stats.get("T_pole_south", 0.0))
        wind_trade_mean = float(stats.get("wind_trade_mean", 0.0))
        wind_midlat_mean = float(stats.get("wind_midlat_mean", 0.0))
        precip_mean = float(stats.get("mean_precip", 0.0))
        precip_min = float(stats.get("precip_min", 0.0))
        precip_max = float(stats.get("precip_max", 0.0))
        precip_equator = float(stats.get("precip_equator", 0.0))
        precip_pole_north = float(stats.get("precip_pole_north", 0.0))
        precip_pole_south = float(stats.get("precip_pole_south", 0.0))
        humidity_mean = float(stats.get("humidity_mean", 0.0))
        humidity_min = float(stats.get("humidity_min", 0.0))
        humidity_max = float(stats.get("humidity_max", 0.0))
        soil_mean = float(stats.get("soil_mean", 0.0))
        soil_min = float(stats.get("soil_min", 0.0))
        soil_max = float(stats.get("soil_max", 0.0))

        self._series["avg_temp"].append(avg_temp)
        self._series["avg_ocean_temp"].append(avg_ocean_temp)
        self._series["wind_mean"].append(wind_mean)
        self._series["wind_min"].append(wind_min)
        self._series["wind_max"].append(wind_max)
        self._series["net_radiation"].append(net_radiation)
        self._series["temp_std"].append(temp_std)
        self._series["circulation_score"].append(circulation_score)
        self._series["ice_cover_fraction"].append(ice_cover_fraction)
        self._series["cloud_mean"].append(cloud_mean)
        self._series["albedo_mean"].append(albedo_mean)
        self._series["wind_div_rms"].append(wind_div_rms)
        self._series["temp_min"].append(temp_min)
        self._series["temp_max"].append(temp_max)
        self._series["temp_equator"].append(temp_equator)
        self._series["temp_pole_north"].append(temp_pole_north)
        self._series["temp_pole_south"].append(temp_pole_south)
        self._series["wind_trade_mean"].append(wind_trade_mean)
        self._series["wind_midlat_mean"].append(wind_midlat_mean)
        self._series["precip_mean"].append(precip_mean)
        self._series["precip_min"].append(precip_min)
        self._series["precip_max"].append(precip_max)
        self._series["precip_equator"].append(precip_equator)
        self._series["precip_pole_north"].append(precip_pole_north)
        self._series["precip_pole_south"].append(precip_pole_south)
        self._series["humidity_mean"].append(humidity_mean)
        self._series["humidity_min"].append(humidity_min)
        self._series["humidity_max"].append(humidity_max)
        self._series["soil_mean"].append(soil_mean)
        self._series["soil_min"].append(soil_min)
        self._series["soil_max"].append(soil_max)

        self._trim_history(total_days)
        self._redraw()

    def _trim_history(self, now_days: float) -> None:
        cutoff = now_days - float(self.history_days)
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            for series in self._series.values():
                series.popleft()

    def _redraw(self) -> None:
        if not self._times:
            return
        t0 = self._times[0]
        xs = [(t - t0) for t in self._times]
        max_x = max(xs[-1], 1e-3)
        xlim = (max(0.0, max_x - float(self.history_days)), max_x)

        for key, line in self._lines.items():
            ys = list(self._series[key])
            line.set_data(xs, ys)
            ax = line.axes
            ax.set_xlim(*xlim)
            ax.relim()
            ax.autoscale_view(scalex=False)

        self.canvas.draw_idle()

    @staticmethod
    def _ocean_mean_temp(state) -> float:
        if state.elevation is None or state.temperature is None:
            return float("nan")
        ocean_mask = _ocean_mask_from_elevation(state.elevation)
        if not np.any(ocean_mask):
            return float("nan")
        return float(np.mean(state.temperature[ocean_mask]))

    @staticmethod
    def _wind_stats(state) -> tuple[float, float, float]:
        if state.wind_u is None or state.wind_v is None:
            return 0.0, 0.0, 0.0
        speed = np.hypot(state.wind_u, state.wind_v)
        return float(np.mean(speed)), float(np.min(speed)), float(np.max(speed))

    @staticmethod
    def _net_radiation(diagnostics) -> float:
        if not diagnostics.component_history:
            return float("nan")
        latest = diagnostics.component_history[-1]
        if "net_radiation_mean" in latest:
            return float(latest["net_radiation_mean"])
        if "net_radiation" in latest:
            return float(latest["net_radiation"])
        return float("nan")


class GraphsController:
    """Controller to manage live graphs on a Tk loop."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        get_state: Callable[[], object],
        diagnostics: object,
        history_days: float = 365.0,
        update_ms: int = 1000,
        toggle_var: Optional[tk.BooleanVar] = None,
    ) -> None:
        self.root = root
        self.get_state = get_state
        self.diagnostics = diagnostics
        self.history_days = float(history_days)
        self.update_ms = int(update_ms)
        self.toggle_var = toggle_var
        self.window: Optional[LiveGraphsWindow] = None
        self._after_id: Optional[str] = None
        self._enabled = False

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self._enabled:
            self._enabled = True
            if self.window is None:
                self.window = LiveGraphsWindow(
                    self.root,
                    history_days=self.history_days,
                    on_close=self._handle_window_close,
                )
            self._schedule()
        elif not enabled and self._enabled:
            self._enabled = False
            self._cancel()
            if self.window is not None:
                self.window.close()
                self.window = None

    def reset(self) -> None:
        if self.window is not None:
            self.window.reset()

    def close(self) -> None:
        self._enabled = False
        self._cancel()
        if self.window is not None:
            self.window.close()
            self.window = None

    def _handle_window_close(self) -> None:
        self._enabled = False
        self._cancel()
        if self.toggle_var is not None:
            self.toggle_var.set(False)

    def _schedule(self) -> None:
        self._cancel()
        self._after_id = self.root.after(self.update_ms, self._tick)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _tick(self) -> None:
        if not self._enabled or self.window is None:
            return
        state = self.get_state()
        self.window.update_from(state, self.diagnostics)
        self._schedule()
