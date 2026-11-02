# ============================================================
#          Integrated MCS Reliability Assessment Tool
# ============================================================
#               (c) 2026 Mikka Kisuule | INESC TEC, Porto
# ============================================================
#
# Standard type catalogue for lines and transformers.
#
# Every line / transformer used by a network must have its parameters stored
# here (or be defined inline in the YAML input file under
# `std_type_definitions:`).  The YAML input files only reference the names.
# ============================================================

from __future__ import annotations

alpha_al = 4.03e-3  # temperature coefficient, aluminium
alpha_cu = 3.93e-3  # temperature coefficient, copper

# --------------------------------------------------------------------------- #
# MV lines
# --------------------------------------------------------------------------- #
LINE_TYPES: dict[str, dict] = {
    # 150 mm2 AAAC overhead line
    "R_150": {
        "r_ohm_per_km": 0.1362,
        "x_ohm_per_km": 0.2222,
        "c_nf_per_km": 221.501,
        "max_i_ka": 0.49,
        "r0_ohm_per_km": 0.4749,
        "x0_ohm_per_km": 1.0128,
        "c0_nf_per_km": 221.501,
        "type": "ol",
        "q_mm2": 150,
        "alpha": alpha_al,
        "voltage_rating": "MV",
    },
    # 100 mm2 AAAC overhead line
    "S_100": {
        "r_ohm_per_km": 0.1774,
        "x_ohm_per_km": 0.3169,
        "c_nf_per_km": 32.094,
        "max_i_ka": 0.395,
        "r0_ohm_per_km": 0.3650,
        "x0_ohm_per_km": 1.5891,
        "c0_nf_per_km": 32.094,
        "type": "ol",
        "q_mm2": 100,
        "alpha": alpha_al,
        "voltage_rating": "MV",
    },
    # 50 mm2 AAAC overhead line
    "T_50": {
        "r_ohm_per_km": 0.2617,
        "x_ohm_per_km": 0.2504,
        "c_nf_per_km": 124.430,
        "max_i_ka": 0.29,
        "r0_ohm_per_km": 0.8975,
        "x0_ohm_per_km": 1.2083,
        "c0_nf_per_km": 124.430,
        "type": "ol",
        "q_mm2": 50,
        "alpha": alpha_al,
        "voltage_rating": "MV",
    },
    # 185 mm2 XLPE copper underground cable
    "P_185": {
        "r_ohm_per_km": 0.1485,
        "x_ohm_per_km": 0.0796,
        "c_nf_per_km": 63.004,
        "max_i_ka": 0.415,
        "r0_ohm_per_km": 1.393,
        "x0_ohm_per_km": 0.2784,
        "c0_nf_per_km": 63.004,
        "type": "cs",
        "q_mm2": 185,
        "alpha": alpha_cu,
        "voltage_rating": "MV",
    },
    # 95 mm2 XLPE copper underground cable
    "Q_95": {
        "r_ohm_per_km": 0.1774,
        "x_ohm_per_km": 0.3169,
        "c_nf_per_km": 46.826,
        "max_i_ka": 0.355,
        "r0_ohm_per_km": 0.3650,
        "x0_ohm_per_km": 1.5891,
        "c0_nf_per_km": 46.826,
        "type": "cs",
        "q_mm2": 100,
        "alpha": alpha_cu,
        "voltage_rating": "MV",
    },
    # ----------------------------------------------------------------------- #
    # LV lines
    # ----------------------------------------------------------------------- #
    # 95 mm2 ABC overhead line
    "H_95": {
        "r_ohm_per_km": 0.3872,
        "x_ohm_per_km": 0.1029,
        "c_nf_per_km": 96.3370,
        "max_i_ka": 0.395,
        "r0_ohm_per_km": 0.0,
        "x0_ohm_per_km": 0.0,
        "c0_nf_per_km": 96.3370,
        "type": "ol",
        "q_mm2": 95,
        "alpha": alpha_al,
        "voltage_rating": "LV",
    },
    # 50 mm2 ABC overhead line
    "J_50": {
        "r_ohm_per_km": 0.4804,
        "x_ohm_per_km": 0.3376,
        "c_nf_per_km": 96.3370,
        "max_i_ka": 0.168,
        "r0_ohm_per_km": 0.0,
        "x0_ohm_per_km": 0.0,
        "c0_nf_per_km": 96.3370,
        "type": "ol",
        "q_mm2": 50,
        "alpha": alpha_al,
        "voltage_rating": "LV",
    },
    # 35 mm2 ABC overhead line
    "K_35": {
        "r_ohm_per_km": 0.6945,
        "x_ohm_per_km": 0.3557,
        "c_nf_per_km": 96.3370,
        "max_i_ka": 0.148,
        "r0_ohm_per_km": 0.0,
        "x0_ohm_per_km": 0.0,
        "c0_nf_per_km": 96.3370,
        "type": "ol",
        "q_mm2": 35,
        "alpha": alpha_al,
        "voltage_rating": "LV",
    },
    # 300 mm2 underground cable
    "A_300": {
        "r_ohm_per_km": 0.1210,
        "x_ohm_per_km": 0.0883,
        "c_nf_per_km": 61.4003,
        "max_i_ka": 0.465,
        "r0_ohm_per_km": 0.7175,
        "x0_ohm_per_km": 0.0508,
        "c0_nf_per_km": 61.4003,
        "type": "cs",
        "q_mm2": 300,
        "alpha": alpha_cu,
        "voltage_rating": "LV",
    },
    # 185 mm2 underground cable
    "B_185": {
        "r_ohm_per_km": 0.1984,
        "x_ohm_per_km": 0.0895,
        "c_nf_per_km": 59.0374,
        "max_i_ka": 0.355,
        "endtemp_degree": 70.0,
        "r0_ohm_per_km": 0.7938,
        "x0_ohm_per_km": 0.0605,
        "c0_nf_per_km": 59.0374,
        "type": "cs",
        "q_mm2": 185,
        "alpha": alpha_cu,
        "voltage_rating": "LV",
    },
    # 120 mm2 underground cable
    "C_120": {
        "r_ohm_per_km": 0.3061,
        "x_ohm_per_km": 0.0895,
        "c_nf_per_km": 58.4495,
        "max_i_ka": 0.280,
        "endtemp_degree": 70.0,
        "r0_ohm_per_km": 1.2245,
        "x0_ohm_per_km": 0.0559,
        "c0_nf_per_km": 58.4495,
        "type": "cs",
        "q_mm2": 120,
        "alpha": alpha_cu,
        "voltage_rating": "LV",
    },
    # 95 mm2 underground cable
    "D_95": {
        "r_ohm_per_km": 0.3872,
        "x_ohm_per_km": 0.0908,
        "c_nf_per_km": 96.3370,
        "max_i_ka": 0.245,
        "r0_ohm_per_km": 1.5488,
        "x0_ohm_per_km": 0.0617,
        "c0_nf_per_km": 96.3370,
        "type": "cs",
        "q_mm2": 95,
        "alpha": alpha_cu,
        "voltage_rating": "LV",
    },
    # 70 mm2 underground cable
    "E_70": {
        "r_ohm_per_km": 0.5360,
        "x_ohm_per_km": 0.0920,
        "c_nf_per_km": 104.3334,
        "max_i_ka": 0.205,
        "endtemp_degree": 70.0,
        "r0_ohm_per_km": 2.1441,
        "x0_ohm_per_km": 0.0629,
        "c0_nf_per_km": 104.3334,
        "type": "cs",
        "q_mm2": 95,
        "alpha": alpha_cu,
        "voltage_rating": "LV",
    },
    # 35 mm2 service cable
    "L_35": {
        "r_ohm_per_km": 1.0297,
        "x_ohm_per_km": 0.0496,
        "c_nf_per_km": 111.8369,
        "max_i_ka": 0.12,
        "r0_ohm_per_km": 4.1188,
        "x0_ohm_per_km": 0.0363,
        "c0_nf_per_km": 111.8369,
        "type": "ol",
        "q_mm2": 35,
        "alpha": alpha_al,
        "voltage_rating": "LV",
    },
    # 25 mm2 service cable
    "M_25": {
        "r_ohm_per_km": 1.4411,
        "x_ohm_per_km": 0.05290,
        "c_nf_per_km": 111.8369,
        "max_i_ka": 0.1,
        "r0_ohm_per_km": 5.7669,
        "x0_ohm_per_km": 0.0363,
        "c0_nf_per_km": 111.8369,
        "type": "ol",
        "q_mm2": 25,
        "alpha": alpha_al,
        "voltage_rating": "LV",
    },
}


def _mv_trafo(sn_mva: float, R: float | None, X: float | None, Xo: float) -> dict:
    """33/11 kV transformer template."""
    data = {
        "sn_mva": sn_mva,
        "vn_hv_kv": 33,
        "vn_lv_kv": 11,
        "vk_percent": 6,
        "vkr_percent": 0.34,
        "pfe_kw": 10.0,
        "i0_percent": 1.0,
        "shift_degree": 30,
        "vector_group": "Yd11",
        "tap_side": "hv",
        "tap_neutral": 0,
        "tap_min": 4,
        "tap_max": -12,
        "tap_step_degree": 0,
        "tap_step_percent": 1.43,
        "tap_phase_shifter": False,
        "vk0_percent": 7.5,
        "vkr0_percent": 1,
        "mag0_percent": 8,
        "mag0_rx": 0.1,
        "si0_hv_partial": 0.9,
        "Xo": Xo,
    }
    if R is not None:
        data["R"] = R
    if X is not None:
        data["X"] = X
    return data


def _lv_trafo(
    sn_mva: float,
    vk_percent: float,
    vkr_percent: float,
    pfe_kw: float,
    R: float | None,
    X: float | None,
    Xo: float,
) -> dict:
    """11 kV / 400 V transformer template."""
    data = {
        "sn_mva": sn_mva,
        "vn_hv_kv": 11,
        "vn_lv_kv": 0.4,
        "vk_percent": vk_percent,
        "vkr_percent": vkr_percent,
        "pfe_kw": pfe_kw,
        "i0_percent": 1.0,
        "shift_degree": 30,
        "vector_group": "Yd11",
        "tap_side": "hv",
        "tap_neutral": 0,
        "tap_min": 5,
        "tap_max": -5,
        "tap_step_degree": 0,
        "tap_step_percent": 2.5,
        "tap_phase_shifter": False,
        "vk0_percent": 7.5,
        "vkr0_percent": 1,
        "mag0_percent": 8,
        "mag0_rx": 0.1,
        "si0_hv_partial": 0.9,
        "Xo": Xo,
    }
    if R is not None:
        data["R"] = R
    if X is not None:
        data["X"] = X
    return data


TRAFO_TYPES: dict[str, dict] = {
    # MV transformers
    "trafo_33_11_15": _mv_trafo(15, R=0.07935, X=1.325, Xo=0.66125),
    "trafo_33_11_5": _mv_trafo(5, R=0.14, X=1.3, Xo=0.8),
    "trafo_33_11_2500": _mv_trafo(2.5, R=0.3609, X=2.8, Xo=1.77),
    # LV transformers
    "trafo_11_400_500": _lv_trafo(
        0.5,
        4.75,
        1.02,
        0.51,
        R=0.003264,
        X=0.014848,
        Xo=0.028,
    ),
    "trafo_11_400_200": _lv_trafo(0.2, 4.75, 1.5, 0.29, R=0.012, X=0.036, Xo=0.028),
    "trafo_11_400_50": _lv_trafo(0.05, 4.5, 2.186, 0.11, R=None, X=None, Xo=0.028),
}

# Combined catalogue consumed by NetworkBuilder
STD_TYPES: dict[str, dict[str, dict]] = {"line": LINE_TYPES, "trafo": TRAFO_TYPES}

__all__ = ["LINE_TYPES", "TRAFO_TYPES", "STD_TYPES", "alpha_al", "alpha_cu"]
