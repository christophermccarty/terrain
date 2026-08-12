"""Keep the product-scope matrix complete as experimental gates are added."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "planet_params.py"
SCOPE_PATH = ROOT / "docs" / "PRODUCT_SCOPE_AND_EXPERIMENTS.md"


def _default_off_planet_params() -> set[str]:
    tree = ast.parse(PARAMS_PATH.read_text(encoding="utf-8"))
    params = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlanetParams"
    )
    return {
        node.target.id
        for node in params.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
    }


def test_every_default_off_planet_param_is_in_the_scope_matrix():
    """A new experimental gate must receive a visible product disposition."""
    scope = SCOPE_PATH.read_text(encoding="utf-8")
    matrix = scope.split("## Default-off gate matrix", 1)[1].split(
        "## Inert numeric trials", 1
    )[0]
    undocumented = sorted(
        name for name in _default_off_planet_params() if f"`{name}`" not in matrix
    )
    assert not undocumented, (
        "Default-off PlanetParams need an explicit disposition in "
        "docs/PRODUCT_SCOPE_AND_EXPERIMENTS.md: " + ", ".join(undocumented)
    )
