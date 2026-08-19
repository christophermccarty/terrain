"""SESAM shortwave radiation -- Appendix A7 of Willeit et al. (2022).

Pure, read-only kernels for the CLIMBER-X/SESAM two-band delta-Eddington
shortwave scheme, transcribed from Appendix A7 of

    Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.:
    GMD 15, 5905-5948 (2022), https://doi.org/10.5194/gmd-15-5905-2022
    (CC-BY 4.0; equations cited below as (A#) against that paper).

This is stage P5's second sub-deliverable (docs/SESAM_GAP_ANALYSIS.md section
7), consuming ``sesam_radiation.py``'s (A6) cloud fraction/height/optical-
thickness as plain array inputs rather than importing that module directly
(same decoupling convention as ``sesam_thermo.py`` accepting P3's AT/Aq as
arguments). The supported PlanetSim pipeline never calls these functions
(the ``PlanetParams.enable_sesam_radiation`` gate is off by default), so
this module has zero default-path climate impact by construction.

**Equations verified directly against the source PDF** (2026-08-19, 500-600
dpi PyMuPDF renders of the appendix pages) and cross-checked (read-only, per
the section 5 licensing policy) against ``src/atm/swr.f90`` and
``src/main/constants.f90`` at github.com/cxesmc/climber-x. Two real paper-vs-
reference discrepancies found and resolved in favor of physics + the running
implementation (same precedent as ``sesam_radiation.py``'s r*/A65 findings):

1. **(A87)/(A88)'s band labels are swapped in the published paper.** As
   printed, (A87) (the water-vapour exponential-absorption formula) is
   subscripted ``wv,uv`` and (A88) (the constant ``I=1``, i.e. no absorption)
   is subscripted ``wv,ir``. This is physically backwards -- water vapour's
   real SW absorption bands (0.94, 1.1, 1.4, 1.9 micron) are all near-infrared;
   it is essentially transparent to visible/UV. The reference implementation
   consistently assigns the exponential formula to the IR band and fixes the
   visible/UV band at 1.0 (``itf_w_ir_s``/``itf_w_vu_s=1`` throughout
   ``swr.f90``, never the reverse). Implemented here as: (A87)'s formula ->
   near-IR transmission, (A88)'s ``1`` -> visible/UV transmission.
2. **(A97)'s cloud-thickness term has a sign error in the published PDF.**
   Printed as ``(1 - e^{+Dcld/Hq})``, which is unbounded (diverges to -inf as
   cloud thickness grows) and inconsistent with the paper's own (A98)/(A99)
   neighbours and the ``f_exp^1``/``f_exp^2`` definitions on the same page,
   all of which use ``e^{-.../Hq}``. The reference implementation computes
   ``exp(-cld_gt/h_q)`` (negative exponent, matching the bounded, physical
   form). Implemented here with the negative exponent.
3. **The aerosol direct-effect surface-albedo modification** (Bauer et al.,
   2008, "eq. 5/6" -- ``l_so4_de`` in the reference implementation) is a
   CLIMBER-X refinement layered on top of the base (A75)-(A78) equations, not
   part of the published appendix algebra itself (which uses the plain
   surface albedo throughout). PlanetSim has no sulfate-aerosol field, so
   this module implements (A69)-(A105) exactly as printed, using the surface
   albedo directly -- consistent with the section 6 policy of building from
   the paper, not importing implementation-only embellishments.
4. **Ozone and cloud shortwave transmission are fixed constants, not fields**
   (A90)-(A93): ``I_O3,vu=0.96``, ``I_O3,ir=1``, ``I_cld,vu=I_cld,ir=0.9``.
   No column-ozone climatology is needed anywhere in this module -- the
   ozone-climatology design question flagged at stage P5's scoping only
   applies to the (A106)-(A117) longwave scheme, not here.
5. **``frac_vu`` (fraction of the solar spectrum in the visible+UV band) is
   a fixed physical constant, 0.45**, read from the reference implementation's
   ``constants.f90`` (not a Table A7 entry -- the published appendix text
   never states its numeric value). ``mu0`` (the effective cosine zenith
   angle for diffuse radiation) is ``1/beta0`` reusing Table A8's already-
   transcribed longwave diffusivity factor (``beta0=1.66``) rather than a new
   constant -- confirmed identical (``cos_zen_o = 1/1.66``) in the reference
   source.
6. ``cos_zenith`` (the direct-beam solar zenith cosine, mu) and
   ``aerosol_optical_thickness``/``aerosol_imaginary_refractive_index``
   (sulfate aerosol loading -- PlanetSim has no aerosol module) are accepted
   as external inputs with no fabricated defaults; a caller with no aerosol
   field should pass zero optical thickness (clean-atmosphere placeholder),
   matching the documented-external-input convention every prior SESAM stage
   uses for genuinely-missing upstream fields.
7. ``humidity_scale_height_m`` (``Hq``) is accepted as a distinct external
   input, not assumed identical to stage P1's RH scale height ``Hr`` (A14) --
   the paper does not state that identity and this module does not invent it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sesam_reference import value as _param

# Fixed physical constants not in the Table A7 XLSX pack (see docstring notes
# 4-5 for provenance of each).
FRAC_VU = 0.45
I_O3_VU = 0.96
I_O3_IR = 1.0
I_CLD_VU = 0.9
I_CLD_IR = 0.9
_EXP_NEG_QUARTER = float(np.exp(-0.25))  # (A96)/(A102) literal constant


def _a7_defaults() -> dict[str, float]:
    """All A7 (shortwave) constants from the published Table A7 pack, plus
    ``mu0`` reused from Table A8's beta0 (docstring note 5)."""
    return {
        "r_sct": _param("A7_shortwave", "r_sct"),
        "g_cld": _param("A7_shortwave", "g_cld"),
        "p1": _param("A7_shortwave", "p1"),
        "p2": _param("A7_shortwave", "p2"),
        "p3": _param("A7_shortwave", "p3"),
        "p4": _param("A7_shortwave", "p4"),
        "alpha1": _param("A7_shortwave", "alpha1"),
        "alpha2": _param("A7_shortwave", "alpha2"),
        "alpha3": _param("A7_shortwave", "alpha3"),
        "gamma1aer": _param("A7_shortwave", "gamma1aer"),
        "gamma2aer": _param("A7_shortwave", "gamma2aer"),
        "D_cld": _param("A7_shortwave", "D_cld"),
        "a1wv_sw": _param("A7_shortwave", "a1wv_sw"),
        "a2wv_sw": _param("A7_shortwave", "a2wv_sw"),
        "b1wv_sw": _param("A7_shortwave", "b1wv_sw"),
        "b2wv_sw": _param("A7_shortwave", "b2wv_sw"),
        "mu0": 1.0 / _param("A8_longwave", "beta0"),
    }


def _check_2d(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (H, W) field")
    return arr


def _matching(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must share a shape")


def column_water_path_g_cm2(column_water_kg_m2: np.ndarray) -> np.ndarray:
    """``W`` [g cm^-2] from column water path in kg m^-2 (P4's ``Qq``).

    ``1 kg/m^2 == 0.1 g/cm^2``.
    """
    qq = _check_2d("column_water_kg_m2", column_water_kg_m2)
    return 0.1 * qq


# ---------------------------------------------------------------------------
# (A79)-(A82) scattering and cloud albedo
# ---------------------------------------------------------------------------


def atmospheric_scattering_albedo(
    cos_zenith: np.ndarray,
    aerosol_optical_thickness: np.ndarray,
    aerosol_imaginary_refractive_index: np.ndarray,
    a7: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(A79)-(A80) atmospheric (Rayleigh + aerosol) scattering albedo.

    ``alb_sct_vu = 1-(1-r_sct)*exp(-mu^p1*(0.55*tau_aer)^p2*f3)``,
    ``alb_sct_ir = 1-exp(-mu^p1*(0.55*tau_aer)^p2*f3)``,
    ``f3 = alpha1 - alpha2*log(1+alpha3*R_aer_im)``.

    Call with ``cos_zenith = mu0`` (the module's diffuse-angle constant) to
    get the "0"-superscript diffuse variant used by (A83)-(A86).
    """
    params = a7 or _a7_defaults()
    mu = _check_2d("cos_zenith", cos_zenith)
    tau = _check_2d("aerosol_optical_thickness", aerosol_optical_thickness)
    r_im = _check_2d("aerosol_imaginary_refractive_index", aerosol_imaginary_refractive_index)
    _matching("cos_zenith", mu, "aerosol_optical_thickness", tau)
    _matching("cos_zenith", mu, "aerosol_imaginary_refractive_index", r_im)

    f1 = np.power(mu, params["p1"])
    f2 = np.power(0.55 * tau, params["p2"])
    f3 = params["alpha1"] - params["alpha2"] * np.log(1.0 + params["alpha3"] * r_im)
    core = np.exp(-f1 * f2 * f3)
    alb_vu = 1.0 - (1.0 - params["r_sct"]) * core
    alb_ir = 1.0 - core
    return alb_vu, alb_ir


def cloud_albedo(
    alb_sct_vu: np.ndarray,
    alb_sct_ir: np.ndarray,
    cos_zenith: np.ndarray,
    cloud_optical_thickness: np.ndarray,
    a7: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(A81)-(A82) cloud albedo from cloud optical thickness.

    ``alb_cld = 1-(1-alb_sct)*exp(-g_cld*tau_cld^p4/mu^p3)``.
    """
    params = a7 or _a7_defaults()
    a_vu = _check_2d("alb_sct_vu", alb_sct_vu)
    a_ir = _check_2d("alb_sct_ir", alb_sct_ir)
    mu = _check_2d("cos_zenith", cos_zenith)
    tau_cld = np.maximum(_check_2d("cloud_optical_thickness", cloud_optical_thickness), 0.0)
    _matching("alb_sct_vu", a_vu, "alb_sct_ir", a_ir)
    _matching("alb_sct_vu", a_vu, "cos_zenith", mu)
    _matching("alb_sct_vu", a_vu, "cloud_optical_thickness", tau_cld)

    b_c = params["g_cld"] / np.power(mu, params["p3"])
    decay = np.exp(-b_c * np.power(tau_cld, params["p4"]))
    alb_cld_vu = 1.0 - (1.0 - a_vu) * decay
    alb_cld_ir = 1.0 - (1.0 - a_ir) * decay
    return alb_cld_vu, alb_cld_ir


# ---------------------------------------------------------------------------
# (A87)-(A89) integral transmission functions: water vapour, aerosol
# ---------------------------------------------------------------------------


def water_vapor_sw_transmission_ir(
    mass_path_g_cm2: np.ndarray, a7: dict[str, float] | None = None
) -> np.ndarray:
    """(A87) water-vapour SW transmission -- IR band (see docstring note 1).

    The visible/UV band is always ``1.0`` (A88, note 1); no function is
    needed for it.
    """
    params = a7 or _a7_defaults()
    m = np.maximum(_check_2d("mass_path_g_cm2", mass_path_g_cm2), 0.0)
    return params["a1wv_sw"] * np.exp(-params["b1wv_sw"] * m) + params["a2wv_sw"] * np.exp(
        -params["b2wv_sw"] * m
    )


def aerosol_sw_transmission(
    mass_path_g_cm2: np.ndarray,
    aerosol_imaginary_refractive_index: np.ndarray,
    a7: dict[str, float] | None = None,
) -> np.ndarray:
    """(A89) aerosol SW transmission, identical for both spectral bands."""
    params = a7 or _a7_defaults()
    m = np.maximum(_check_2d("mass_path_g_cm2", mass_path_g_cm2), 0.0)
    r_im = _check_2d("aerosol_imaginary_refractive_index", aerosol_imaginary_refractive_index)
    _matching("mass_path_g_cm2", m, "aerosol_imaginary_refractive_index", r_im)
    return np.exp(-params["gamma1aer"] * m * np.power(r_im, params["gamma2aer"]))


# ---------------------------------------------------------------------------
# (A94)-(A105) absorber mass paths (shared structure, water and aerosol)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbsorberMassPaths:
    """Six mass-path variants sharing one (A94)-(A105) structure.

    ``cs``/``cld`` feed the TOA planetary-albedo transmission products;
    ``cs1``/``cs2`` and ``cld1``/``cld2`` are the surface direct/diffuse
    split used by (A83)-(A86).
    """

    cs: np.ndarray
    cs1: np.ndarray
    cs2: np.ndarray
    cld: np.ndarray
    cld1: np.ndarray
    cld2: np.ndarray


def absorber_mass_paths(
    path_content: np.ndarray,
    cos_zenith: np.ndarray,
    cloud_top_height_m: np.ndarray,
    cloud_geometric_thickness_m: np.ndarray,
    humidity_scale_height_m: np.ndarray,
    a7: dict[str, float] | None = None,
) -> AbsorberMassPaths:
    """(A94)-(A105): pass ``path_content=W`` for water (A94-A99) or
    ``path_content=0.55*tau_aer`` for aerosol (A100-A105) -- the two are the
    literal same formula in the paper, differing only in that substitution.
    """
    params = a7 or _a7_defaults()
    x = _check_2d("path_content", path_content)
    mu = _check_2d("cos_zenith", cos_zenith)
    hcld = _check_2d("cloud_top_height_m", cloud_top_height_m)
    dcld = _check_2d("cloud_geometric_thickness_m", cloud_geometric_thickness_m)
    hq = _check_2d("humidity_scale_height_m", humidity_scale_height_m)
    for name, arr in (
        ("cos_zenith", mu), ("cloud_top_height_m", hcld),
        ("cloud_geometric_thickness_m", dcld), ("humidity_scale_height_m", hq),
    ):
        _matching("path_content", x, name, arr)

    mu0 = params["mu0"]
    icos = 1.0 / mu + 1.0 / mu0
    hq_safe = np.maximum(hq, 1e-6)

    cs = icos * x
    cs1 = x / mu
    cs2 = cs1 + (1.0 - _EXP_NEG_QUARTER) * (2.0 / mu0) * x

    exp_hc_hq = np.exp(-hcld / hq_safe)
    # (A97) uses -Dcld/Hq (docstring note 2: corrects the published +Dcld/Hq).
    cld = exp_hc_hq * (icos + (1.0 - np.exp(-dcld / hq_safe))) * x

    f_exp1 = exp_hc_hq - np.exp(-(hcld + dcld) / hq_safe)
    f_exp2 = 1.0 - exp_hc_hq / mu0
    cld1 = (exp_hc_hq / mu + f_exp1 + f_exp2) * x
    cld2 = cld1 + (f_exp1 + 2.0 * f_exp2) * x

    return AbsorberMassPaths(cs=cs, cs1=cs1, cs2=cs2, cld=cld, cld1=cld1, cld2=cld2)


# ---------------------------------------------------------------------------
# (A75)-(A78) planetary albedo, (A83)-(A86) surface transmission
# ---------------------------------------------------------------------------


def _layered_albedo(alb_layer: np.ndarray, alb_surface: np.ndarray) -> np.ndarray:
    """Two-stream adding-method combinator shared by (A75)-(A78):
    ``alb_layer + (1-alb_layer)^2*alb_surface/(1-alb_layer*alb_surface)``.
    """
    return alb_layer + np.power(1.0 - alb_layer, 2) * alb_surface / (1.0 - alb_layer * alb_surface)


def planetary_albedo(
    alb_layer_vu: np.ndarray,
    alb_layer_ir: np.ndarray,
    surface_albedo_vu: np.ndarray,
    surface_albedo_ir: np.ndarray,
    water_transmission_ir: np.ndarray,
    aerosol_transmission: np.ndarray,
    ozone_transmission_vu: float,
    ozone_transmission_ir: float,
    cloud_transmission: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(A75)-(A78) planetary albedo for one sky condition (clear or cloudy).

    Pass ``alb_layer=alb_sct`` for the clear-sky pair (A75)-(A76) or
    ``alb_layer=alb_cld`` for the cloudy pair (A77)-(A78); the combinator is
    identical, only the transmission-product terms differ (the visible/UV
    water term is always 1.0, docstring note 1, so it is omitted here).
    """
    a_vu = _check_2d("alb_layer_vu", alb_layer_vu)
    a_ir = _check_2d("alb_layer_ir", alb_layer_ir)
    s_vu = _check_2d("surface_albedo_vu", surface_albedo_vu)
    s_ir = _check_2d("surface_albedo_ir", surface_albedo_ir)
    itf_w_ir = _check_2d("water_transmission_ir", water_transmission_ir)
    itf_a = _check_2d("aerosol_transmission", aerosol_transmission)
    _matching("alb_layer_vu", a_vu, "alb_layer_ir", a_ir)
    _matching("alb_layer_vu", a_vu, "surface_albedo_vu", s_vu)
    _matching("alb_layer_vu", a_vu, "surface_albedo_ir", s_ir)
    _matching("alb_layer_vu", a_vu, "water_transmission_ir", itf_w_ir)
    _matching("alb_layer_vu", a_vu, "aerosol_transmission", itf_a)

    alb_atm_vu = _layered_albedo(a_vu, s_vu) * itf_a * ozone_transmission_vu * cloud_transmission
    alb_atm_ir = (
        _layered_albedo(a_ir, s_ir) * itf_w_ir * itf_a * ozone_transmission_ir * cloud_transmission
    )
    return alb_atm_vu, alb_atm_ir


def surface_transmission(
    alb_layer_direct_vu: np.ndarray,
    alb_layer_direct_ir: np.ndarray,
    alb_layer_diffuse_vu: np.ndarray,
    alb_layer_diffuse_ir: np.ndarray,
    surface_albedo_vu: np.ndarray,
    surface_albedo_ir: np.ndarray,
    itf_d1_ir: np.ndarray,
    itf_d2_ir: np.ndarray,
    itf_aerosol_d1: np.ndarray,
    itf_aerosol_d2: np.ndarray,
    ozone_transmission_vu: float,
    ozone_transmission_ir: float,
    cloud_transmission: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(A83)-(A86) atmospheric integral transmission to the surface.

    Pass ``alb_layer_direct=alb_layer_diffuse=alb_sct``/``alb_sct_0`` for the
    clear-sky pair (A83)-(A84), or both equal to ``alb_cld`` for the cloudy
    pair (A85)-(A86) (docstring: the paper uses one cloud albedo throughout,
    no separate diffuse variant). The visible/UV water term is always 1.0
    (note 1), so ``itf_d1``/``itf_d2`` here are IR-band water transmission
    only; aerosol transmission (identical both bands, A89) is a separate
    argument since it is not always 1 in the visible band.
    """
    a_dir_vu = _check_2d("alb_layer_direct_vu", alb_layer_direct_vu)
    a_dir_ir = _check_2d("alb_layer_direct_ir", alb_layer_direct_ir)
    a_dif_vu = _check_2d("alb_layer_diffuse_vu", alb_layer_diffuse_vu)
    a_dif_ir = _check_2d("alb_layer_diffuse_ir", alb_layer_diffuse_ir)
    s_vu = _check_2d("surface_albedo_vu", surface_albedo_vu)
    s_ir = _check_2d("surface_albedo_ir", surface_albedo_ir)
    itf_w_d1 = _check_2d("itf_d1_ir", itf_d1_ir)
    itf_w_d2 = _check_2d("itf_d2_ir", itf_d2_ir)
    itf_a_d1 = _check_2d("itf_aerosol_d1", itf_aerosol_d1)
    itf_a_d2 = _check_2d("itf_aerosol_d2", itf_aerosol_d2)
    for name, arr in (
        ("alb_layer_direct_ir", a_dir_ir), ("alb_layer_diffuse_vu", a_dif_vu),
        ("alb_layer_diffuse_ir", a_dif_ir), ("surface_albedo_vu", s_vu),
        ("surface_albedo_ir", s_ir), ("itf_d1_ir", itf_w_d1), ("itf_d2_ir", itf_w_d2),
        ("itf_aerosol_d1", itf_a_d1), ("itf_aerosol_d2", itf_a_d2),
    ):
        _matching("alb_layer_direct_vu", a_dir_vu, name, arr)

    def _one_band(a_dir, a_dif, s, itf_water_d1, itf_water_d2, i_o3):
        term1 = (1.0 - a_dir) * (1.0 - s) * itf_water_d1 * itf_a_d1 * i_o3 * cloud_transmission
        term2 = (
            (1.0 - a_dir) * s * a_dif * (1.0 - s) / (1.0 - a_dif * s)
            * itf_water_d2 * itf_a_d2 * i_o3 * cloud_transmission
        )
        return term1 + term2

    itf_atm_vu = _one_band(a_dir_vu, a_dif_vu, s_vu, 1.0, 1.0, ozone_transmission_vu)
    itf_atm_ir = _one_band(a_dir_ir, a_dif_ir, s_ir, itf_w_d1, itf_w_d2, ozone_transmission_ir)
    return itf_atm_vu, itf_atm_ir


# ---------------------------------------------------------------------------
# (A69)-(A74) clear/cloudy weighted-mean combination
# ---------------------------------------------------------------------------


def band_combine(value_vu: np.ndarray, value_ir: np.ndarray) -> np.ndarray:
    """``frac_vu*value_vu + (1-frac_vu)*value_ir`` -- the two-band combiner
    used throughout (A69)-(A86) wherever visible/UV and IR results merge.
    """
    v_vu = _check_2d("value_vu", value_vu)
    v_ir = _check_2d("value_ir", value_ir)
    _matching("value_vu", v_vu, "value_ir", v_ir)
    return FRAC_VU * v_vu + (1.0 - FRAC_VU) * v_ir


def sky_combine(
    clear_sky_value: np.ndarray, cloudy_value: np.ndarray, cloud_fraction: np.ndarray
) -> np.ndarray:
    """(A69)/(A70): ``fcld*cloudy + (1-fcld)*clear``."""
    cs = _check_2d("clear_sky_value", clear_sky_value)
    cld = _check_2d("cloudy_value", cloudy_value)
    f = np.clip(_check_2d("cloud_fraction", cloud_fraction), 0.0, 1.0)
    _matching("clear_sky_value", cs, "cloudy_value", cld)
    _matching("clear_sky_value", cs, "cloud_fraction", f)
    return f * cld + (1.0 - f) * cs


# ---------------------------------------------------------------------------
# Full-pipeline assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShortwaveRadiation:
    """Assembled (A69)/(A70) shortwave fluxes plus the intermediate albedos,
    exposed for diagnostics/testing rather than only the two final fields."""

    toa_upward_w_m2: np.ndarray
    surface_downward_w_m2: np.ndarray
    clear_sky_toa_albedo_vu: np.ndarray
    clear_sky_toa_albedo_ir: np.ndarray
    cloudy_toa_albedo_vu: np.ndarray
    cloudy_toa_albedo_ir: np.ndarray


def shortwave_radiation(
    incoming_toa_w_m2: np.ndarray,
    cos_zenith: np.ndarray,
    cloud_fraction: np.ndarray,
    cloud_top_height_m: np.ndarray,
    cloud_optical_thickness: np.ndarray,
    cloud_geometric_thickness_m: np.ndarray,
    column_water_kg_m2: np.ndarray,
    humidity_scale_height_m: np.ndarray,
    surface_albedo_vu: np.ndarray,
    surface_albedo_ir: np.ndarray,
    aerosol_optical_thickness: np.ndarray,
    aerosol_imaginary_refractive_index: np.ndarray,
    a7: dict[str, float] | None = None,
) -> ShortwaveRadiation:
    """Full (A69)-(A105) pipeline: TOA-upward and surface-downward SW flux.

    ``cos_zenith`` is the direct-beam solar zenith cosine (docstring note 6);
    ``aerosol_optical_thickness``/``aerosol_imaginary_refractive_index`` pass
    0.0 for a clean-atmosphere placeholder absent a real aerosol field (same
    note); ``humidity_scale_height_m`` is a distinct input from stage P1's Hr
    (docstring note 7); ``surface_albedo_vu``/``_ir`` may be the same
    broadband value as a documented simplification until PlanetSim has a
    per-band albedo field.
    """
    params = a7 or _a7_defaults()
    mu0 = np.full_like(_check_2d("cos_zenith", cos_zenith), params["mu0"])

    w = column_water_path_g_cm2(column_water_kg_m2)
    b_ar = 0.55 * _check_2d("aerosol_optical_thickness", aerosol_optical_thickness)

    alb_sct_vu, alb_sct_ir = atmospheric_scattering_albedo(
        cos_zenith, aerosol_optical_thickness, aerosol_imaginary_refractive_index, a7=params
    )
    alb_sct_vu_0, alb_sct_ir_0 = atmospheric_scattering_albedo(
        mu0, aerosol_optical_thickness, aerosol_imaginary_refractive_index, a7=params
    )
    alb_cld_vu, alb_cld_ir = cloud_albedo(
        alb_sct_vu, alb_sct_ir, cos_zenith, cloud_optical_thickness, a7=params
    )

    mw = absorber_mass_paths(
        w, cos_zenith, cloud_top_height_m, cloud_geometric_thickness_m,
        humidity_scale_height_m, a7=params,
    )
    ma = absorber_mass_paths(
        b_ar, cos_zenith, cloud_top_height_m, cloud_geometric_thickness_m,
        humidity_scale_height_m, a7=params,
    )

    itf_w_ir_cs = water_vapor_sw_transmission_ir(mw.cs, a7=params)
    itf_w_ir_cld = water_vapor_sw_transmission_ir(mw.cld, a7=params)
    itf_w_ir_cs1 = water_vapor_sw_transmission_ir(mw.cs1, a7=params)
    itf_w_ir_cs2 = water_vapor_sw_transmission_ir(mw.cs2, a7=params)
    itf_w_ir_cld1 = water_vapor_sw_transmission_ir(mw.cld1, a7=params)
    itf_w_ir_cld2 = water_vapor_sw_transmission_ir(mw.cld2, a7=params)

    itf_a_cs = aerosol_sw_transmission(ma.cs, aerosol_imaginary_refractive_index, a7=params)
    itf_a_cld = aerosol_sw_transmission(ma.cld, aerosol_imaginary_refractive_index, a7=params)
    itf_a_cs1 = aerosol_sw_transmission(ma.cs1, aerosol_imaginary_refractive_index, a7=params)
    itf_a_cs2 = aerosol_sw_transmission(ma.cs2, aerosol_imaginary_refractive_index, a7=params)
    itf_a_cld1 = aerosol_sw_transmission(ma.cld1, aerosol_imaginary_refractive_index, a7=params)
    itf_a_cld2 = aerosol_sw_transmission(ma.cld2, aerosol_imaginary_refractive_index, a7=params)

    alb_atm_cs_vu, alb_atm_cs_ir = planetary_albedo(
        alb_sct_vu, alb_sct_ir, surface_albedo_vu, surface_albedo_ir,
        itf_w_ir_cs, itf_a_cs, I_O3_VU, I_O3_IR,
    )
    alb_atm_cld_vu, alb_atm_cld_ir = planetary_albedo(
        alb_cld_vu, alb_cld_ir, surface_albedo_vu, surface_albedo_ir,
        itf_w_ir_cld, itf_a_cld, I_O3_VU, I_O3_IR, cloud_transmission=I_CLD_VU,
    )
    # I_CLD_VU == I_CLD_IR (A92==A93); planetary_albedo takes one scalar
    # cloud_transmission applied to both bands, which is exact here.

    sw_top_up_cs = band_combine(
        _check_2d("incoming_toa_w_m2", incoming_toa_w_m2) * alb_atm_cs_vu,
        _check_2d("incoming_toa_w_m2", incoming_toa_w_m2) * alb_atm_cs_ir,
    )
    sw_top_up_cld = band_combine(
        _check_2d("incoming_toa_w_m2", incoming_toa_w_m2) * alb_atm_cld_vu,
        _check_2d("incoming_toa_w_m2", incoming_toa_w_m2) * alb_atm_cld_ir,
    )
    toa_upward = sky_combine(sw_top_up_cs, sw_top_up_cld, cloud_fraction)

    itf_atm_cs_vu, itf_atm_cs_ir = surface_transmission(
        alb_sct_vu, alb_sct_ir, alb_sct_vu_0, alb_sct_ir_0,
        surface_albedo_vu, surface_albedo_ir,
        itf_w_ir_cs1, itf_w_ir_cs2, itf_a_cs1, itf_a_cs2,
        I_O3_VU, I_O3_IR,
    )
    itf_atm_cld_vu, itf_atm_cld_ir = surface_transmission(
        alb_cld_vu, alb_cld_ir, alb_cld_vu, alb_cld_ir,
        surface_albedo_vu, surface_albedo_ir,
        itf_w_ir_cld1, itf_w_ir_cld2, itf_a_cld1, itf_a_cld2,
        I_O3_VU, I_O3_IR, cloud_transmission=I_CLD_VU,
    )

    sw_sur_cs = band_combine(
        _check_2d("incoming_toa_w_m2", incoming_toa_w_m2) * itf_atm_cs_vu,
        _check_2d("incoming_toa_w_m2", incoming_toa_w_m2) * itf_atm_cs_ir,
    )
    sw_sur_cld = band_combine(
        _check_2d("incoming_toa_w_m2", incoming_toa_w_m2) * itf_atm_cld_vu,
        _check_2d("incoming_toa_w_m2", incoming_toa_w_m2) * itf_atm_cld_ir,
    )
    surface_downward = sky_combine(sw_sur_cs, sw_sur_cld, cloud_fraction)

    return ShortwaveRadiation(
        toa_upward_w_m2=toa_upward,
        surface_downward_w_m2=surface_downward,
        clear_sky_toa_albedo_vu=alb_atm_cs_vu,
        clear_sky_toa_albedo_ir=alb_atm_cs_ir,
        cloudy_toa_albedo_vu=alb_atm_cld_vu,
        cloudy_toa_albedo_ir=alb_atm_cld_ir,
    )


__all__ = [
    "FRAC_VU", "I_O3_VU", "I_O3_IR", "I_CLD_VU", "I_CLD_IR",
    "column_water_path_g_cm2",
    "atmospheric_scattering_albedo",
    "cloud_albedo",
    "water_vapor_sw_transmission_ir",
    "aerosol_sw_transmission",
    "AbsorberMassPaths",
    "absorber_mass_paths",
    "planetary_albedo",
    "surface_transmission",
    "band_combine",
    "sky_combine",
    "ShortwaveRadiation",
    "shortwave_radiation",
]
