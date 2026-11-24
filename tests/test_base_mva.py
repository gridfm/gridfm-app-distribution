"""
Per-network MVA base (plan item 1.1): ``data.baseMVA`` holds one value per
network, it is checked against each network's network_info.txt before any
normalisation, and each network's normaliser converts the admittances and
shunts with its own base.
"""

import math
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gridfm_app_dist.datasets.globals import BS, GS, YFF_TT_R, YFT_TF_I  # noqa: E402
from gridfm_app_dist.datasets.network_info import (  # noqa: E402
    BaseMVAError,
    check_base_mva,
    configured_base_mva,
    main as network_info_main,
)
from gridfm_app_dist.io.param_handler import NestedNamespace  # noqa: E402

BB = ("bus", "connects", "bus")


def _args(networks, base_mva="unset", **extra):
    data = {"networks": networks, "normalization": "HeteroDataMVANormalizer"}
    data.update(extra)
    if base_mva != "unset":
        data["baseMVA"] = base_mva
    return NestedNamespace(data=data, training={"batch_size": 2})


def _info(tmp_path, network, base, raw=False):
    folder = tmp_path / network / ("raw" if raw else "")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "network_info.txt").write_text(
        f"network_name: {network}\nbase_mva: {base}\ncontingency_components:\n  buses: 3\n",
    )


# ------------------------------------------------------------------ config format
def test_configured_base_mva_forms():
    assert configured_base_mva(_args(["a", "b"], [1, 100])) == [1.0, 100.0]
    assert configured_base_mva(_args(["a", "b"], {"b": 100, "a": 1})) == [1.0, 100.0]
    assert configured_base_mva(_args(["a"], 0.01)) == [0.01]


@pytest.mark.parametrize(
    "networks, base",
    [
        (["a", "b"], 100),  # one value for several networks
        (["a", "b"], [1]),  # wrong length
        (["a", "b"], {"a": 1}),  # mapping without b
        (["a"], "unset"),  # missing
        (["a"], [0]),  # not positive
        (["a"], ["x"]),  # not a number
    ],
)
def test_configured_base_mva_rejects(networks, base):
    with pytest.raises(BaseMVAError):
        configured_base_mva(_args(networks, base))


# ------------------------------------------------------------------ network_info check
def test_check_matches_network_info(tmp_path):
    _info(tmp_path, "a", 1.0)
    _info(tmp_path, "b", 100.0, raw=True)  # raw/ folder is searched too
    assert check_base_mva(_args(["a", "b"], [1, 100]), str(tmp_path)) == [1.0, 100.0]


def test_check_reports_every_problem(tmp_path):
    _info(tmp_path, "a", 1.0)
    _info(tmp_path, "b", 100.0)
    with pytest.raises(BaseMVAError) as err:
        check_base_mva(_args(["a", "b", "c"], [100, 100, 1]), str(tmp_path))
    msg = str(err.value)
    assert "a: data.baseMVA = 100" in msg and "says base_mva = 1" in msg
    assert "c: no network_info.txt" in msg
    assert "b:" not in msg


def test_check_can_be_disabled_for_legacy_checkpoints(tmp_path):
    _info(tmp_path, "a", 1.0)
    args = _args(["a"], [100], check_base_mva=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert check_base_mva(args, str(tmp_path)) == [100.0]
    assert any("NOT checked" in str(w.message) for w in caught)


def test_preflight_cli(tmp_path, capsys):
    _info(tmp_path, "a", 1.0)
    _info(tmp_path, "b", 100.0)
    good = tmp_path / "good.yaml"
    good.write_text("data:\n  networks: [a, b]\n  baseMVA: [1.0, 100.0]\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  networks: [a, b]\n  baseMVA: 100\n")
    assert network_info_main(["--config", str(good), "--data_path", str(tmp_path)]) == 0
    assert network_info_main(["--config", str(bad), "--data_path", str(tmp_path)]) == 1
    assert "baseMVA: [1, 100]" in capsys.readouterr().out  # suggested line


def test_datamodule_stops_before_normalisation(tmp_path):
    pytest.importorskip("lightning")
    from gridfm_app_dist.datasets.hetero_powergrid_datamodule import (
        LitGridHeteroDataModule,
    )

    _info(tmp_path, "a", 1.0)
    with pytest.raises(BaseMVAError):
        LitGridHeteroDataModule(_args(["a"], [100]), data_dir=str(tmp_path))
    dm = LitGridHeteroDataModule(_args(["a"], [1.0]), data_dir=str(tmp_path))
    assert dm.base_mva == [1.0]


# ------------------------------------------------------------------ normalisers
def _sample():
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["bus"].x = torch.ones(2, 17)
    data["bus"].y = torch.ones(2, 7)
    data["gen"].x = torch.ones(1, 7)
    data["gen"].y = torch.ones(1, 7)
    data[BB].edge_index = torch.tensor([[0, 1], [1, 0]])
    data[BB].edge_attr = torch.ones(2, 13)
    return data


def _fitted(base_mva, fitted_base=0.5):
    from gridfm_app_dist.io.param_handler import load_normalizer
    import gridfm_app_dist.datasets.normalizers  # noqa: F401  (registers them)

    args = _args(["a", "b"], [base_mva, 7.0])
    norm = load_normalizer(args, base_mva=base_mva)
    assert norm.baseMVA_orig == base_mva
    norm.fit_from_dict(
        {
            "baseMVA_orig": torch.tensor(base_mva),
            "baseMVA": torch.tensor(fitted_base),
            "vn_kv_max": torch.tensor(1.0),
        },
    )
    return norm


def test_each_network_uses_its_own_base():
    for base in (1.0, 100.0, 0.01):
        data = _sample()
        _fitted(base).transform(data)
        factor = base / 0.5
        ea = data.edge_attr_dict[BB]
        assert torch.allclose(
            ea[:, YFF_TT_R : YFT_TF_I + 1],
            torch.full((2, 4), factor),
        )
        assert torch.allclose(data.x_dict["bus"][:, GS], torch.full((2,), factor))
        assert torch.allclose(data.x_dict["bus"][:, BS], torch.full((2,), factor))
        assert float(data.baseMVA_orig) == pytest.approx(base)


def test_normaliser_defaults():
    from gridfm_app_dist.datasets.normalizers import HeteroDataMVANormalizer

    assert HeteroDataMVANormalizer(_args(["a"], [0.4])).baseMVA_orig == 0.4
    # several networks and no explicit base: unset until given
    norm = HeteroDataMVANormalizer(_args(["a", "b"], [1, 100]))
    assert norm.baseMVA_orig is None
    norm.baseMVA, norm.vn_kv_max = 0.5, 1.0
    with pytest.raises(ValueError, match="MVA base"):
        norm.transform(_sample())


def test_saved_stats_with_another_base_are_rejected():
    from gridfm_app_dist.datasets.normalizers import HeteroDataMVANormalizer

    norm = HeteroDataMVANormalizer(_args(["a"], [1.0]))
    with pytest.raises(ValueError, match="configured base"):
        norm.fit_from_dict(
            {
                "baseMVA_orig": torch.tensor(100.0),  # stats of a legacy run
                "baseMVA": torch.tensor(0.5),
                "vn_kv_max": torch.tensor(1.0),
            },
        )


# ------------------------------------------------------------------ units audit
# A two-bus network solved in physical units: the line admittance is in p.u. on
# the network's own base sn_mva (as in the raw data), powers and flows in MW /
# Mvar, angles in degrees. After normalisation every power must be in p.u. of
# the fitted base B, and the injections and flows computed from the normalised
# admittances and voltages must reproduce those powers, for any sn_mva.
# (The saved statistics are float32, hence rtol 1e-6.)
Y_LINE = complex(2.0, -8.0)
V_TRUE = (complex(1.0, 0.0), 0.98 * complex(math.cos(-0.02), math.sin(-0.02)))


def _physical_sample(sn_mva):
    from torch_geometric.data import HeteroData
    from gridfm_app_dist.datasets.globals import (
        MAX_QG_H,
        MIN_QG_H,
        P_E,
        PD_H,
        PG_BUS_H,
        Q_E,
        QD_H,
        QG_H,
        RATE_A,
        VA_H,
        VM_H,
        VN_KV,
        YFF_TT_I,
        YFF_TT_R,
        YFT_TF_I,
        YFT_TF_R,
    )

    v1, v2 = V_TRUE
    # flows (MW/Mvar) from each end, S_ij = sn * V_i conj(y (V_i - V_j))
    s12 = sn_mva * v1 * (Y_LINE * (v1 - v2)).conjugate()
    s21 = sn_mva * v2 * (Y_LINE * (v2 - v1)).conjugate()

    data = HeteroData()
    x = torch.zeros(2, 17, dtype=torch.float64)
    y = torch.zeros(2, 7, dtype=torch.float64)
    for i, v in enumerate(V_TRUE):
        vm, va_deg = abs(v), math.degrees(math.atan2(v.imag, v.real))
        x[i, VM_H] = y[i, VM_H] = vm
        x[i, VA_H] = y[i, VA_H] = va_deg
    x[:, VN_KV] = 0.4
    # bus 1: slack injecting s12; bus 2: load -s21 (S_net = S_inj at both buses)
    y[0, PG_BUS_H], y[0, QG_H], x[0, QG_H] = s12.real, s12.imag, s12.imag
    y[1, PD_H], y[1, QD_H] = -s21.real, -s21.imag
    x[1, PD_H], x[1, QD_H] = -s21.real, -s21.imag
    x[0, MIN_QG_H], x[0, MAX_QG_H] = -5.0 * sn_mva, 5.0 * sn_mva
    data["bus"].x, data["bus"].y = x, y
    data["gen"].x = torch.zeros(0, 7, dtype=torch.float64)
    data["gen"].y = torch.zeros(0, 1, dtype=torch.float64)

    ea = torch.zeros(2, 13, dtype=torch.float64)
    ea[:, YFF_TT_R], ea[:, YFF_TT_I] = Y_LINE.real, Y_LINE.imag  # Yff (fwd), Ytt (rev)
    ea[:, YFT_TF_R], ea[:, YFT_TF_I] = -Y_LINE.real, -Y_LINE.imag  # Yft, Ytf
    ea[0, P_E], ea[0, Q_E] = s12.real, s12.imag
    ea[1, P_E], ea[1, Q_E] = s21.real, s21.imag
    ea[:, RATE_A] = 2.0 * sn_mva
    data[BB].edge_index = torch.tensor([[0, 1], [1, 0]])
    data[BB].edge_attr = ea
    data[BB].y = ea[:, [P_E, Q_E]].clone()
    data["gen", "connected_to", "bus"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    data["scenario_id"] = torch.tensor([0])
    return data, s12, s21


def _normaliser(kind, sn_mva, base):
    from gridfm_app_dist.datasets.normalizers import (
        HeteroDataMVANormalizer,
        HeteroDataPerSampleMVANormalizer,
    )

    args = _args(["a"], [sn_mva])
    stats = {
        "baseMVA_orig": torch.tensor(sn_mva),
        "baseMVA": torch.tensor(base),
        "vn_kv_max": torch.tensor(0.4),
    }
    if kind == "global":
        norm = HeteroDataMVANormalizer(args)
    else:
        norm = HeteroDataPerSampleMVANormalizer(args)
        stats = dict(
            stats,
            scenarios=torch.tensor([0]),
            baseMVA=torch.tensor([base]),
            vn_kv_max=torch.tensor([0.4]),
        )
    norm.fit_from_dict(stats)
    return norm


@pytest.mark.parametrize("kind", ["global", "per_sample"])
@pytest.mark.parametrize("sn_mva", [0.01, 1.0, 100.0])
def test_every_quantity_is_on_the_fitted_base(kind, sn_mva):
    from gridfm_app_dist.datasets.globals import (
        MAX_QG_H,
        P_E,
        PD_H,
        PG_BUS_H,
        Q_E,
        QD_H,
        QG_H,
        RATE_A,
        VA_H,
        VM_H,
        YFF_TT_I,
        YFF_TT_R,
        YFT_TF_I,
        YFT_TF_R,
    )

    base = 0.37 * sn_mva  # fitted power base (MW), deliberately != sn_mva
    data, s12, s21 = _physical_sample(sn_mva)
    _normaliser(kind, sn_mva, base).transform(data)
    x, y = data.x_dict["bus"], data.y_dict["bus"]
    ea = data.edge_attr_dict[BB]

    def close(t, v):
        return torch.allclose(
            t,
            torch.as_tensor(v, dtype=t.dtype),
            rtol=1e-6,
            atol=1e-12,
        )

    # powers: MW / Mvar -> p.u. of the fitted base
    assert close(y[1, PD_H], -s21.real / base) and close(y[1, QD_H], -s21.imag / base)
    assert close(y[0, PG_BUS_H], s12.real / base)  # was multiplied (per-sample bug)
    assert close(y[0, QG_H], s12.imag / base) and close(x[0, QG_H], s12.imag / base)
    assert close(x[0, MAX_QG_H], 5.0 * sn_mva / base)
    assert close(
        ea[:, P_E],
        torch.tensor([s12.real, s21.real], dtype=torch.float64) / base,
    )
    assert close(
        ea[:, Q_E],
        torch.tensor([s12.imag, s21.imag], dtype=torch.float64) / base,
    )
    assert close(ea[:, RATE_A], 2.0 * sn_mva / base)
    assert close(
        data[BB].y[:, 0],
        torch.tensor([s12.real, s21.real], dtype=torch.float64) / base,
    )
    assert close(data.baseMVA_orig, sn_mva)

    # physics with the normalised admittances reproduces the normalised powers
    vm, va = y[:, VM_H], y[:, VA_H]
    assert close(
        va,
        torch.tensor([math.atan2(v.imag, v.real) for v in V_TRUE], dtype=torch.float64),
    )
    V = vm * torch.exp(1j * va)
    yff = torch.complex(ea[:, YFF_TT_R], ea[:, YFF_TT_I])
    yft = torch.complex(ea[:, YFT_TF_R], ea[:, YFT_TF_I])
    src, dst = data[BB].edge_index
    s_flow = V[src] * torch.conj(
        yff * V[src] + yft * V[dst],
    )  # per edge, from its source
    assert torch.allclose(s_flow.real, ea[:, P_E], rtol=1e-6)
    assert torch.allclose(s_flow.imag, ea[:, Q_E], rtol=1e-6)
    s_net = torch.complex(y[:, PG_BUS_H] - y[:, PD_H], y[:, QG_H] - y[:, QD_H])
    assert torch.allclose(s_flow, s_net[src], rtol=1e-6)  # bus balance (one line each)


@pytest.mark.parametrize("kind", ["global", "per_sample"])
def test_inverse_transform_restores_powers(kind):
    from gridfm_app_dist.datasets.globals import P_E, PD_H, PG_BUS_H, QG_H, RATE_A

    sn_mva, base = 1.0, 0.37
    data, _, _ = _physical_sample(sn_mva)
    ref, _, _ = _physical_sample(sn_mva)
    norm = _normaliser(kind, sn_mva, base)
    norm.transform(data)
    norm.inverse_transform(data)
    for t, r in (
        (data.y_dict["bus"], ref.y_dict["bus"]),
        (
            data.edge_attr_dict[BB][:, [P_E, RATE_A]],
            ref.edge_attr_dict[BB][:, [P_E, RATE_A]],
        ),
        (data[BB].y, ref[BB].y),
    ):
        cols = (
            [PD_H, QG_H, PG_BUS_H] if t.dim() == 2 and t.size(1) == 7 else slice(None)
        )
        assert torch.allclose(t[:, cols], r[:, cols], rtol=1e-9)
