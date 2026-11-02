"""
Bus-bus switches must be exported as graph edges, and every switch state must
reach the model's input flags, so that the BUS_STATUS label equals the
REF-reachability the model can derive from its inputs (0 conflicts).

The frame-level test runs anywhere; the generation test needs pandapower.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from contingency_data_gen.generation import datakit_extractors as dke


def test_reachability_from_flags_counts_conflicts():
    bus = pd.DataFrame(
        {
            "scenario": 0,
            "bus": [10, 11, 12],
            "REF": [1, 0, 0],
            "PV": 0,
            "BUS_BASE_STATUS": 1,
            "BUS_CONTINGENCY": 0,
            "BUS_STATUS": [1, 1, 1],
        },
    )
    branch = pd.DataFrame(
        {
            "scenario": 0,
            "from_bus": [10, 11],
            "to_bus": [11, 12],
            "br_status": [1, 1],
            "BRANCH_BASE_STATUS": [1, 1],
            "BRANCH_CONTINGENCY": [0, 0],
        },
    )
    assert dke.reachability_from_flags(bus, branch).tolist() == [True, True, True]
    assert dke.count_reachability_conflicts(bus, branch)["conflict_buses"] == 0
    # drop the 11-12 edge (e.g. a switch that was never exported): bus 12 conflicts
    res = dke.count_reachability_conflicts(bus, branch.iloc[:1])
    assert res["conflict_buses"] == 1 and res["conflict_reach0_label1"] == 1


def test_unexplained_branch_outages_are_reported():
    bus = pd.DataFrame(
        {
            "scenario": 0,
            "bus": [10, 11, 12],
            "BUS_BASE_STATUS": 1,
            "BUS_CONTINGENCY": [0, 0, 1],
        },
    )
    branch = pd.DataFrame(
        {
            "scenario": 0,
            "idx": [0, 1, 2, 3],
            "from_bus": [10, 11, 10, 10],
            "to_bus": [11, 12, 12, 11],
            # 0: healthy & on; 1: off, touches the hit bus 12 (explained);
            # 2: off, contingency (explained); 3: off for no visible reason
            "br_status": [1, 0, 0, 0],
            "BRANCH_BASE_STATUS": [1, 1, 1, 1],
            "BRANCH_CONTINGENCY": [0, 0, 1, 0],
        },
    )
    res = dke.count_unexplained_branch_outages(bus, branch)
    assert res["unexplained_off_branches"] == 1
    assert res["examples"] == [{"scenario": 0, "idx": 3}]


def _lv_net(pp):
    """MV/LV feeder with a fuse (bus-bus), a line switch, a bus-bus CB and an open tie."""
    net = pp.create_empty_network(sn_mva=1.0)
    mv = pp.create_bus(net, vn_kv=20.0)
    lv = [pp.create_bus(net, vn_kv=0.4) for _ in range(7)]
    pp.create_ext_grid(net, mv)
    pp.create_transformer(net, mv, lv[0], std_type="0.4 MVA 20/0.4 kV")
    fuse = pp.create_switch(net, lv[0], lv[1], et="b", closed=True, type="LS")
    l0 = pp.create_line(net, lv[1], lv[2], 0.1, "NAYY 4x150 SE")
    l1 = pp.create_line(net, lv[2], lv[3], 0.1, "NAYY 4x150 SE")
    pp.create_switch(net, lv[3], l1, et="l", closed=True, type="LBS")
    cb = pp.create_switch(net, lv[3], lv[4], et="b", closed=True, type="CB")
    pp.create_line(net, lv[4], lv[5], 0.1, "NAYY 4x150 SE")
    pp.create_line(net, lv[5], lv[6], 0.1, "NAYY 4x150 SE")
    tie = pp.create_switch(net, lv[2], lv[6], et="b", closed=False, type="LBS")
    for b in lv[2:]:
        pp.create_load(net, b, p_mw=0.01, q_mvar=0.002)
    return net, (fuse, cb, tie), l0


@pytest.mark.slow
def test_switch_edges_and_zero_conflicts():
    pp = pytest.importorskip(
        "pandapower",
        reason="pandapower is required for generation",
    )
    from contingency_data_gen.generation.scenario_data_gen import ScenarioGenerator

    net, (fuse, cb, tie), _ = _lv_net(pp)
    g = ScenarioGenerator(
        network_file_path=None,
        flag=None,
        net=net,
        output_dir=None,
        name="switch_test",
        prepare_output_dir=False,
    )
    g.load_network_from_excel()
    bus_frames, branch_frames, sid = [], [], 0
    for n in (1, 2):
        scenarios = [list(c) for c in itertools.combinations(g.all_components, n)]
        half = len(scenarios) // 2
        for chunk in (
            scenarios[:half],
            scenarios[half:],
        ):  # two batches: no stale state
            b, _, br = g.run_batch_contingency_analysis(n, chunk, sid)
            sid += len(chunk)
            bus_frames.append(b)
            branch_frames.append(br)
    bus = pd.concat(bus_frames, ignore_index=True)
    branch = pd.concat(branch_frames, ignore_index=True)

    n_bb = int((net.switch.et == "b").sum())
    assert branch.groupby("scenario").size().unique().tolist() == [
        len(net.line) + len(net.trafo) + n_bb,
    ]
    sw_rows = branch[branch["idx"] >= len(net.line) + len(net.trafo)]
    y_cols = [
        "r",
        "x",
        "b",
        "Yff_r",
        "Yff_i",
        "Yft_r",
        "Yft_i",
        "Ytf_r",
        "Ytf_i",
        "Ytt_r",
        "Ytt_i",
    ]
    assert np.allclose(sw_rows[y_cols].to_numpy(), 0.0)
    # the normally-open tie is in the graph but unavailable in the base case
    base = branch[branch["scenario"] == 0].set_index("idx")["BRANCH_BASE_STATUS"]
    tie_row = (
        len(net.line)
        + len(net.trafo)
        + list(net.switch[net.switch.et == "b"].index).index(tie)
    )
    assert int(base.loc[tie_row]) == 0
    # base flags never change between scenarios / batches
    assert bus.groupby("bus")["BUS_BASE_STATUS"].nunique().max() == 1
    assert branch.groupby("idx")["BRANCH_BASE_STATUS"].nunique().max() == 1

    res = dke.count_reachability_conflicts(bus, branch, max_scenarios=None)
    assert res["conflict_buses"] == 0, res
    # pandapower must not switch off branches only because their area is dead
    br = dke.count_unexplained_branch_outages(bus, branch, max_scenarios=None)
    assert br["unexplained_off_branches"] == 0, br
