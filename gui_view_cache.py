"""Shared cache state for expensive GUI-only view layers."""
from __future__ import annotations

WIND_CACHE = {"key": None, "u": None, "v": None}
OCEAN_CURRENT_CACHE = {
    "key": None,
    "u": None,
    "v": None,
    "computed_at": 0.0,
}
PRECIP_VIEW_CACHE = {"key": None, "P": None}
OCEAN_CURRENT_REFRESH_SEC = 2.0


def invalidate_gui_view_caches() -> None:
    """Invalidate every cached GUI overlay and its refresh timestamp."""
    WIND_CACHE.update({"key": None, "u": None, "v": None})
    OCEAN_CURRENT_CACHE.update(
        {"key": None, "u": None, "v": None, "computed_at": 0.0}
    )
    PRECIP_VIEW_CACHE.update({"key": None, "P": None})
