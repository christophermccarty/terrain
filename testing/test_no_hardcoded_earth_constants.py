"""Guard against re-introducing hardcoded Earth-only constants in physics modules.

This is a direct response to two bugs found in the 2026-07-03 audit pass: a
hardcoded 6371 km storm-geometry radius, and a `% 365` cache key that aliased
Mars day 400 onto day 35 (wrong season). Both were only catchable by reading
every line by hand. This test would have caught them mechanically.

Scope is deliberately narrow: it flags the exact constants known to be
Earth-only physical values (radius, surface pressure, mean temperature,
gravity) plus the specific `% 365` hardcoded-year-length anti-pattern. It does
NOT flag every `365.0` multiplication — many of those are legitimate per-day
to per-year unit conversions for diagnostics/display, not planet-day-length
assumptions, and would make this test too noisy to be useful.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCANNED_FILES = [
    "atmosphere.py",
    "ocean.py",
    "carbon_cycle.py",
    "temperature.py",
    "simulate.py",
]

SUSPECT_PATTERNS = [
    (re.compile(r"\b6371(\.0)?\b"), "Earth radius (km) -- use pp.radius_m / 1000.0"),
    (re.compile(r"\b1013\.25\b"), "Earth surface pressure (hPa) -- use pp.surface_pressure_pa / 100.0"),
    (re.compile(r"\b288\.15\b"), "Earth mean surface temperature (K) -- source from PlanetParams, not a literal"),
    (re.compile(r"\b9\.81\b"), "Earth surface gravity (m/s^2) -- use pp.surface_gravity"),
    (re.compile(r"%\s*365(\.\d+)?\b"), "hardcoded 365-day modulo -- use pp.orbital_period_days"),
    # Added 2026-08-02 with ACCURACY_AUDIT.md C3's fix. 8848 (Everest) was
    # hardcoded in 4 places across 3 modules with 3 different formulas, and the
    # 6.5 K/km lapse rate at 5 call sites, so Mars terrain silently got Earth's
    # height ceiling and Earth's cooling-per-km.
    (re.compile(r"\b8848(\.0)?\b"), "Earth max elevation (m, Everest) -- use pp.max_elevation_km"),
    (re.compile(r"\b8748(\.0)?\b"), "Earth max elevation minus the 100 m lowland branch -- derive from pp.max_elevation_km"),
    (re.compile(r"\b8\.848\b"), "Earth max elevation (km, Everest) -- use pp.max_elevation_km"),
    (re.compile(r"(?<![\w.])6\.5(?![\w.])"), "Earth lapse rate (K/km) -- use pp.lapse_rate_k_per_km"),
]

# (filename, substring required on the matching line) -- reviewed, legitimate
# uses. A hit is allowlisted only if the substring appears on that exact line,
# so moving/editing the line will re-trigger review.
ALLOWLIST_LINE_SUBSTRINGS = [
    # Function-default kwarg, always overridden by pp.radius_m at every call site.
    ("atmosphere.py", "planet_radius_km: float = 6371.0"),
    # Function-default kwarg, always overridden by pp.surface_pressure_pa / 100.0
    # at every call site inside simulate.py.
    ("atmosphere.py", "surface_pressure_hpa: float = 1013.25"),
    ("simulate.py", "surface_pressure_hpa=1013.25"),
    # Function-default kwarg, overridden from pp.max_elevation_km at every
    # physics call site (simulate.py's orographic-cooling block and
    # temperature.py's overlay). Same reviewed pattern as planet_radius_km above.
    ("temperature.py", "max_elevation_km: float = 8.848"),
]


def _code_lines(path: Path):
    """Yield (line_no, full_line, code_part) for real code lines only.

    Skips blank lines, `#` comments, and -- since 2026-08-02 -- the interiors of
    triple-quoted strings. Docstrings routinely *describe* these constants in
    prose ("Earth's default is Everest (8.848 km)"), which is documentation, not
    a hardcoded physics value; without this the guard forces an ever-growing
    allowlist of prose lines and gets ignored. Simple toggle rather than a full
    parse: it handles the one-delimiter-per-line convention this codebase uses,
    including single-line `\"\"\"...\"\"\"` docstrings.
    """
    in_doc = False
    doc_delim = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if in_doc:
            if doc_delim in line:
                in_doc = False
                doc_delim = None
            continue
        if not stripped or stripped.startswith("#"):
            continue
        for delim in ('"""', "'''"):
            if delim in stripped:
                # Opens a docstring unless it also closes on the same line.
                if stripped.count(delim) == 1:
                    in_doc = True
                    doc_delim = delim
                break
        if in_doc:
            continue
        code_part = line.split("#", 1)[0]  # ignore trailing inline comments
        # Drop any single-line triple-quoted string content on this line too.
        code_part = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', "", code_part)
        yield line_no, line, code_part


def test_no_hardcoded_earth_constants():
    violations = []
    for filename in SCANNED_FILES:
        path = REPO_ROOT / filename
        for line_no, full_line, code_part in _code_lines(path):
            for pattern, hint in SUSPECT_PATTERNS:
                if not pattern.search(code_part):
                    continue
                if any(
                    filename == allow_file and allow_substr in full_line
                    for allow_file, allow_substr in ALLOWLIST_LINE_SUBSTRINGS
                ):
                    continue
                violations.append(f"{filename}:{line_no}: {full_line.strip()!r} -- {hint}")

    assert not violations, (
        "Hardcoded Earth-specific constant(s) found. Source the value from "
        "PlanetParams instead, or -- if this really is a reviewed, legitimate "
        "case (e.g. a default kwarg always overridden by the caller) -- add a "
        "justified entry to ALLOWLIST_LINE_SUBSTRINGS in this file:\n"
        + "\n".join(violations)
    )
