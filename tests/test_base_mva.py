"""
Per-network MVA base (plan item 1.1): ``data.baseMVA`` holds one value per
network, it is checked against each network's network_info.txt before any
normalisation, and each network's normaliser converts the admittances and
shunts with its own base.
"""

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
        assert torch.allclose(ea[:, YFF_TT_R : YFT_TF_I + 1], torch.full((2, 4), factor))
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
