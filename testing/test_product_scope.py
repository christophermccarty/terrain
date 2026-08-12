"""Keep the experimental-gate inventory aligned with PlanetParams defaults."""
from __future__ import annotations

from pathlib import Path

from planet_params import EARTH, PlanetParams


ROOT = Path(__file__).resolve().parents[1]
SCOPE_DOC = ROOT / "docs" / "PRODUCT_SCOPE_AND_EXPERIMENTS.md"


def test_product_scope_documents_every_inert_earth_control():
    """Every false/zero Earth control needs an explicit product disposition."""
    text = SCOPE_DOC.read_text(encoding="utf-8")
    default_off_gates = sorted(name for name in PlanetParams.__dataclass_fields__ if getattr(EARTH, name) is False)
    default_zero_controls = sorted(
        name
        for name in PlanetParams.__dataclass_fields__
        if isinstance(getattr(EARTH, name), (int, float))
        and not isinstance(getattr(EARTH, name), bool)
        and getattr(EARTH, name) == 0
    )
    documented_controls = default_off_gates + default_zero_controls

    undocumented = [name for name in documented_controls if f"`{name}`" not in text]
    assert not undocumented, (
        "Document every false/zero Earth control in "
        f"{SCOPE_DOC.relative_to(ROOT)}: {', '.join(undocumented)}"
    )
