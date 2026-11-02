"""
datakit_extractors.py
=====================

Separate per-column extraction functions that map a solved pandapower network
(under a given N-k contingency scenario) onto the exact column layout used by
`gridfm-datakit` parquet output.

Reference schema (from gridfm-datakit `utils/column_names.py`,
`process/process_network.py` and `utils/power_balance.py`):

    bus_data.parquet    : scenario, load_scenario_idx, bus, Pd, Qd, Pg, Qg,
                          Vm, Va, PQ, PV, REF, vn_kv, min_vm_pu, max_vm_pu, GS, BS
    gen_data.parquet    : scenario, load_scenario_idx, idx, bus, p_mw, q_mvar,
                          min_p_mw, max_p_mw, min_q_mvar, max_q_mvar,
                          cp0_eur, cp1_eur_per_mw, cp2_eur_per_mw2,
                          in_service, is_slack_gen
    branch_data.parquet : scenario, load_scenario_idx, idx, from_bus, to_bus,
                          pf, qf, pt, qt, r, x, b,
                          Yff_r, Yff_i, Yft_r, Yft_i, Ytf_r, Ytf_i, Ytt_r, Ytt_i,
                          tap, shift, ang_min, ang_max, rate_a, br_status

The dataloader only reads the base datakit columns, while the VLD task additionally consumes the
topology-status supervision columns):

    bus     : + BUS_BASE_STATUS, BUS_CONTINGENCY, BUS_STATUS
    branch  : + BRANCH_BASE_STATUS, BRANCH_CONTINGENCY, BRANCH_STATUS

Branch rows are lines, then trafos, then bus-bus switches (``net.switch`` rows
with ``et == "b"``). A bus-bus switch is exported as a zero-impedance edge
(r = x = b = 0, so every Y element is 0) so the model sees the connection, and
its base status / contingency flag / energisation label are carried in the VLD
columns exactly like a line's. Without these rows the buses a closed switch
joins are disconnected in the graph, while the REF-reachability label
(``ScenarioGenerator._compute_reachable_buses``) treats them as connected.
Line and trafo switches (``et`` in ``"l"``/``"t"``) are not separate edges:
their state is folded into the status columns of the line / trafo they sit on.

Design notes
------------
* Every column is produced by its own small function (`bus_col_*`,
  `gen_col_*`, `branch_col_*`) so each mapping is auditable and testable in isolation,
  The public assemblers (`extract_bus_frame`, `extract_gen_frame`, `extract_branch_frame`) simply
  call the per-column functions in schema order.
* Sign / unit conventions follow datakit:
    - Pd, Qd, Pg, Qg, p_mw, q_mvar, pf/qf/pt/qt are in MW / MVAr.
    - GS, BS are the pandapower per-bus shunt conductance/susceptance
      normalised by the system base (`/ baseMVA`), matching datakit
      (`net.buses[:, GS] / net.baseMVA`).
    - r, x, b, tap, shift come from the MATPOWER-format per-branch record in
      `net._ppc["branch"]` (system per-unit, radians for shift), i.e. the same
      representation datakit derives its admittances from.
    - Yff/Yft/Ytf/Ytt use the datakit `compute_branch_admittances` equations.
* All extractors read from a network that has already been solved with
  `pp.runpp`. Voltage / flow results come from `res_*` tables; static
  parameters come from the element tables and the internal ppc.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# MATPOWER branch-matrix column indices (pandapower stores branches in
# `net._ppc["branch"]` using the MATPOWER layout).
# ---------------------------------------------------------------------------
BR_F_BUS = 0
BR_T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
BR_RATE_A = 5
BR_TAP = 8
BR_SHIFT = 9  # degrees in the ppc matrix
BR_STATUS = 10
BR_ANGMIN = 11
BR_ANGMAX = 12


# ===========================================================================
# Admittance helper
# ===========================================================================
def compute_branch_admittances(r, x, b, tap_mag, shift_rad):
    """Vectorised Yff/Yft/Ytf/Ytt from series/shunt params and complex tap.

    Yff = (y_series + y_sh) / t2
    Yft = -y_series / conj(tap)
    Ytf = -y_series / tap
    Ytt =  y_series + y_sh

    with y_series = 1/(r + jx), y_sh = j*b/2, tap = tap_mag*exp(j*shift),
    t2 = tap_mag**2. `shift_rad` must be in radians.
    """
    r = np.asarray(r, dtype=float)
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)
    tap_mag = np.asarray(tap_mag, dtype=float)
    shift_rad = np.asarray(shift_rad, dtype=float)

    z_series = r + 1j * x
    with np.errstate(divide="ignore", invalid="ignore"):
        y_series = np.where(np.abs(z_series) > 0, 1.0 / z_series, 0.0 + 0.0j)

    tap = tap_mag * (np.cos(shift_rad) + 1j * np.sin(shift_rad))
    t2 = np.where(tap_mag != 0, tap_mag**2, 1.0)
    y_sh = 1j * (b / 2.0)

    Yff = (y_series + y_sh) / t2
    Yft = np.where(tap != 0, -y_series / np.conj(tap), 0.0 + 0.0j)
    Ytf = np.where(tap != 0, -y_series / tap, 0.0 + 0.0j)
    Ytt = y_series + y_sh
    return Yff, Yft, Ytf, Ytt


# ===========================================================================
# BUS: per-column extraction functions
# ===========================================================================
# Each function receives the solved `net` plus any precomputed context it
# needs and returns a 1-D array/list aligned with `net.bus.index` order.


def bus_index(net):
    """Bus identifier column (`bus`)."""
    return np.asarray(net.bus.index, dtype=np.int64)


def bus_col_Pd(net):
    """Active load per bus in MW (`Pd`), only in-service loads."""
    out = np.zeros(len(net.bus), dtype=float)
    if len(net.load):
        m = net.load.in_service.values
        agg = net.load.loc[m].groupby("bus")["p_mw"].sum()
        for pos, bus_idx in enumerate(net.bus.index):
            out[pos] = float(agg.get(bus_idx, 0.0))
    return out


def bus_col_Qd(net):
    """Reactive load per bus in MVAr (`Qd`), only in-service loads."""
    out = np.zeros(len(net.bus), dtype=float)
    if len(net.load):
        m = net.load.in_service.values
        agg = net.load.loc[m].groupby("bus")["q_mvar"].sum()
        for pos, bus_idx in enumerate(net.bus.index):
            out[pos] = float(agg.get(bus_idx, 0.0))
    return out


def bus_col_Pg(net):
    """Active generation per bus in MW (`Pg`): gens + ext_grid results."""
    out = np.zeros(len(net.bus), dtype=float)
    pos_of = {bus_idx: pos for pos, bus_idx in enumerate(net.bus.index)}
    if len(net.gen):
        m = net.gen.in_service.values
        p = net.res_gen.p_mw.values
        for bus_idx, pv, keep in zip(net.gen.bus.values, p, m):
            if keep and bus_idx in pos_of:
                out[pos_of[bus_idx]] += float(pv)
    if len(net.ext_grid):
        for bus_idx, pv in zip(net.ext_grid.bus.values, net.res_ext_grid.p_mw.values):
            if bus_idx in pos_of:
                out[pos_of[bus_idx]] += float(pv)
    return out


def bus_col_Qg(net):
    """Reactive generation per bus in MVAr (`Qg`): gens + ext_grid results."""
    out = np.zeros(len(net.bus), dtype=float)
    pos_of = {bus_idx: pos for pos, bus_idx in enumerate(net.bus.index)}
    if len(net.gen):
        m = net.gen.in_service.values
        q = net.res_gen.q_mvar.values
        for bus_idx, qv, keep in zip(net.gen.bus.values, q, m):
            if keep and bus_idx in pos_of:
                out[pos_of[bus_idx]] += float(qv)
    if len(net.ext_grid):
        for bus_idx, qv in zip(net.ext_grid.bus.values, net.res_ext_grid.q_mvar.values):
            if bus_idx in pos_of:
                out[pos_of[bus_idx]] += float(qv)
    return out


def bus_col_Vm(net):
    """Voltage magnitude in p.u. (`Vm`). NaN results -> 0.0 (de-energised)."""
    vm = net.res_bus.vm_pu.reindex(net.bus.index).values.astype(float)
    return np.nan_to_num(vm, nan=0.0)


def bus_col_Va(net):
    """Voltage angle in degrees (`Va`). NaN results -> 0.0."""
    va = net.res_bus.va_degree.reindex(net.bus.index).values.astype(float)
    return np.nan_to_num(va, nan=0.0)


def _bus_type_masks(net):
    """(PQ, PV, REF) boolean masks aligned with net.bus.index order."""
    ref_buses = set(net.ext_grid.bus.values.tolist()) if len(net.ext_grid) else set()
    pv_buses = set()
    if len(net.gen):
        m = net.gen.in_service.values
        pv_buses = set(net.gen.bus.values[m].tolist())
    pq, pv, ref = [], [], []
    for bus_idx in net.bus.index:
        is_ref = bus_idx in ref_buses
        is_pv = (bus_idx in pv_buses) and not is_ref
        is_pq = not (is_ref or is_pv)
        pq.append(int(is_pq))
        pv.append(int(is_pv))
        ref.append(int(is_ref))
    return np.array(pq), np.array(pv), np.array(ref)


def bus_col_PQ(net):
    """One-hot PQ bus-type indicator (`PQ`)."""
    return _bus_type_masks(net)[0]


def bus_col_PV(net):
    """One-hot PV bus-type indicator (`PV`)."""
    return _bus_type_masks(net)[1]


def bus_col_REF(net):
    """One-hot slack/reference bus-type indicator (`REF`)."""
    return _bus_type_masks(net)[2]


def bus_col_vn_kv(net):
    """Base voltage in kV (`vn_kv`)."""
    return net.bus.vn_kv.values.astype(float)


def bus_col_min_vm_pu(net):
    """Lower voltage limit in p.u. (`min_vm_pu`). Falls back to 0.9."""
    if "min_vm_pu" in net.bus.columns:
        return net.bus.min_vm_pu.fillna(0.9).values.astype(float)
    return np.full(len(net.bus), 0.9, dtype=float)


def bus_col_max_vm_pu(net):
    """Upper voltage limit in p.u. (`max_vm_pu`). Falls back to 1.1."""
    if "max_vm_pu" in net.bus.columns:
        return net.bus.max_vm_pu.fillna(1.1).values.astype(float)
    return np.full(len(net.bus), 1.1, dtype=float)


def bus_col_GS(net, base_mva):
    """Shunt conductance normalised by base MVA (`GS`).

    pandapower stores shunts in a separate `net.shunt` table (p_mw at 1 p.u.).
    Datakit uses `net.buses[:, GS] / baseMVA`; the pandapower shunt `p_mw`
    at 1 p.u. voltage equals the MATPOWER `GS` value, so we aggregate shunt
    p_mw per bus and divide by base.
    """
    out = np.zeros(len(net.bus), dtype=float)
    if hasattr(net, "shunt") and len(net.shunt):
        m = (
            net.shunt.in_service.values
            if "in_service" in net.shunt.columns
            else np.ones(len(net.shunt), bool)
        )
        agg = net.shunt.loc[m].groupby("bus")["p_mw"].sum()
        for pos, bus_idx in enumerate(net.bus.index):
            out[pos] = float(agg.get(bus_idx, 0.0)) / base_mva
    return out


def bus_col_BS(net, base_mva):
    """Shunt susceptance normalised by base MVA (`BS`).

    Uses `-q_mvar` because pandapower defines a capacitive shunt with negative
    q_mvar while MATPOWER `BS` is positive for capacitive support.
    """
    out = np.zeros(len(net.bus), dtype=float)
    if hasattr(net, "shunt") and len(net.shunt):
        m = (
            net.shunt.in_service.values
            if "in_service" in net.shunt.columns
            else np.ones(len(net.shunt), bool)
        )
        agg = net.shunt.loc[m].groupby("bus")["q_mvar"].sum()
        for pos, bus_idx in enumerate(net.bus.index):
            out[pos] = -float(agg.get(bus_idx, 0.0)) / base_mva
    return out


# --- VLD extension columns (bus) ------------------------------------------
def bus_col_BUS_BASE_STATUS(net, base_bus_status):
    """Healthy-in-base indicator per bus (VLD supervision context)."""
    return np.array(
        [int(base_bus_status.get(int(b), 1)) for b in net.bus.index],
        dtype=np.int64,
    )


def bus_col_BUS_CONTINGENCY(net, contingency_buses):
    """1 where the bus is a directly-failed contingency element (VLD)."""
    return np.array(
        [int(int(b) in contingency_buses) for b in net.bus.index],
        dtype=np.int64,
    )


def bus_col_BUS_STATUS(net, reachable_buses):
    """Post-contingency energisation label from REF reachability (VLD target)."""
    return np.array(
        [int(bool(reachable_buses.get(int(b), False))) for b in net.bus.index],
        dtype=np.int64,
    )


# ===========================================================================
# GEN: per-column extraction functions
# ===========================================================================


def gen_col_idx(net):
    """Sequential generator index (`idx`)."""
    return np.arange(len(net.gen), dtype=np.int64)


def gen_col_bus(net):
    """Generator terminal bus (`bus`)."""
    return net.gen.bus.values.astype(np.int64)


def gen_col_p_mw(net):
    """Dispatched active power in MW (`p_mw`); 0 for out-of-service gens."""
    p = net.res_gen.p_mw.reindex(net.gen.index).values.astype(float)
    p = np.nan_to_num(p, nan=0.0)
    p[~net.gen.in_service.values] = 0.0
    return p


def gen_col_q_mvar(net):
    """Dispatched reactive power in MVAr (`q_mvar`); 0 for out-of-service gens."""
    q = net.res_gen.q_mvar.reindex(net.gen.index).values.astype(float)
    q = np.nan_to_num(q, nan=0.0)
    q[~net.gen.in_service.values] = 0.0
    return q


def gen_col_min_p_mw(net):
    """Active-power lower limit (`min_p_mw`)."""
    if "min_p_mw" in net.gen.columns:
        return net.gen.min_p_mw.fillna(0.0).values.astype(float)
    return np.zeros(len(net.gen), dtype=float)


def gen_col_max_p_mw(net):
    """Active-power upper limit (`max_p_mw`)."""
    if "max_p_mw" in net.gen.columns:
        return net.gen.max_p_mw.fillna(0.0).values.astype(float)
    return np.zeros(len(net.gen), dtype=float)


def gen_col_min_q_mvar(net):
    """Reactive-power lower limit (`min_q_mvar`)."""
    if "min_q_mvar" in net.gen.columns:
        return net.gen.min_q_mvar.fillna(0.0).values.astype(float)
    return np.zeros(len(net.gen), dtype=float)


def gen_col_max_q_mvar(net):
    """Reactive-power upper limit (`max_q_mvar`)."""
    if "max_q_mvar" in net.gen.columns:
        return net.gen.max_q_mvar.fillna(0.0).values.astype(float)
    return np.zeros(len(net.gen), dtype=float)


def _gen_cost_matrix(net):
    """Return (cp0, cp1, cp2) arrays aligned with net.gen index order.

    pandapower keeps polynomial costs in `net.poly_cost` keyed by
    (element, et). Datakit orders coefficients as cp0=constant, cp1=linear,
    cp2=quadratic. We read cp0_eur/cp1_eur_per_mw/cp2_eur_per_mw2 from
    poly_cost when present, else 0.
    """
    n = len(net.gen)
    cp0 = np.zeros(n, dtype=float)
    cp1 = np.zeros(n, dtype=float)
    cp2 = np.zeros(n, dtype=float)
    if hasattr(net, "poly_cost") and len(net.poly_cost):
        pc = net.poly_cost
        gen_rows = pc[pc.et == "gen"]
        for pos, gen_idx in enumerate(net.gen.index):
            row = gen_rows[gen_rows.element == gen_idx]
            if len(row):
                r0 = row.iloc[0]
                cp0[pos] = float(r0.get("cp0_eur", 0.0) or 0.0)
                cp1[pos] = float(r0.get("cp1_eur_per_mw", 0.0) or 0.0)
                cp2[pos] = float(r0.get("cp2_eur_per_mw2", 0.0) or 0.0)
    return cp0, cp1, cp2


def gen_col_cp0_eur(net):
    """Constant cost coefficient (`cp0_eur`)."""
    return _gen_cost_matrix(net)[0]


def gen_col_cp1_eur_per_mw(net):
    """Linear cost coefficient (`cp1_eur_per_mw`)."""
    return _gen_cost_matrix(net)[1]


def gen_col_cp2_eur_per_mw2(net):
    """Quadratic cost coefficient (`cp2_eur_per_mw2`)."""
    return _gen_cost_matrix(net)[2]


def gen_col_in_service(net):
    """In-service flag (`in_service`)."""
    return net.gen.in_service.values.astype(np.int64)


def gen_col_is_slack_gen(net):
    """1 for generators sitting on the reference (slack) bus (`is_slack_gen`)."""
    ref_buses = set(net.ext_grid.bus.values.tolist()) if len(net.ext_grid) else set()
    return np.array(
        [int(b in ref_buses) for b in net.gen.bus.values],
        dtype=np.int64,
    )


# ===========================================================================
# BRANCH: per-column extraction functions
# ===========================================================================
# Branch here means a *physical* branch (line, trafo or bus-bus switch), one
# row per element, matching datakit's per-branch layout. We build a unified
# branch table by concatenating lines, then trafos, then bus-bus switches. Line
# and trafo params come from net._ppc["branch"]; pandapower fuses the buses of a
# closed bus-bus switch instead of giving it a ppc row, so switch rows are
# appended here as zero-impedance edges (r = x = b = 0 -> all Y elements 0).


def bus_bus_switches(net):
    """Bus-bus switches (``et == "b"``) in ``net.switch`` index order."""
    if len(net.switch) == 0:
        return net.switch.iloc[0:0]
    return net.switch[net.switch.et == "b"].sort_index()


def switch_is_closed_and_in_service(net, switch_idx):
    """Electrical state of one switch: closed and (if the column exists) in service."""
    row = net.switch.loc[switch_idx]
    in_service = bool(row["in_service"]) if "in_service" in net.switch.columns else True
    return in_service and bool(row["closed"])


def build_branch_static(net, base_mva):
    """Assemble the static per-branch table (params + admittances).

    Returns a dict of arrays keyed by branch column name plus helper keys
    `_n_line` / `_n_trafo` / `_n_switch` and `_switch_idx`.
    Row order: all lines (in net.line order), then all trafos (in net.trafo
    order) — this matches how pandapower stacks them into ppc["branch"] — then
    all bus-bus switches (in net.switch order), which have no ppc row.

    Switch rows: from_bus = switch.bus, to_bus = switch.element, r = x = b = 0
    (every admittance element is therefore 0), tap = 1, shift = 0, the default
    angle limits, rate_a = 0, and br_status = 1 when the switch is closed and in
    service in this scenario (0 otherwise, like an out-of-service line).
    """
    ppc_branch = net._ppc["branch"]
    n_line = len(net.line)
    n_trafo = len(net.trafo)
    switches = bus_bus_switches(net)
    n_switch = len(switches)
    n_phys = n_line + n_trafo  # rows that have a ppc["branch"] counterpart
    n_branch = n_phys + n_switch

    from_bus = np.zeros(n_branch, dtype=np.int64)
    to_bus = np.zeros(n_branch, dtype=np.int64)
    r = np.zeros(n_branch, dtype=float)
    x = np.zeros(n_branch, dtype=float)
    b = np.zeros(n_branch, dtype=float)
    tap = np.ones(n_branch, dtype=float)
    shift_deg = np.zeros(n_branch, dtype=float)
    ang_min = np.full(n_branch, -360.0, dtype=float)
    ang_max = np.full(n_branch, 360.0, dtype=float)
    rate_a = np.zeros(n_branch, dtype=float)
    br_status = np.zeros(n_branch, dtype=np.int64)

    # Physical from/to buses come from the element tables (real bus ids).
    for i, (_, row) in enumerate(net.line.iterrows()):
        from_bus[i] = int(row.from_bus)
        to_bus[i] = int(row.to_bus)
    for j, (_, row) in enumerate(net.trafo.iterrows()):
        from_bus[n_line + j] = int(row.hv_bus)
        to_bus[n_line + j] = int(row.lv_bus)
    for k, (sw_idx, row) in enumerate(switches.iterrows()):
        from_bus[n_phys + k] = int(row.bus)
        to_bus[n_phys + k] = int(row.element)
        # r, x, b, shift, rate_a stay 0, tap stays 1, angle limits stay default.
        br_status[n_phys + k] = int(switch_is_closed_and_in_service(net, sw_idx))

    # Electrical params (system p.u.) from the MATPOWER-format ppc branch matrix.
    # Only line and trafo rows have a ppc counterpart; never read past them
    # (ppc rows after the trafos are impedances / xwards, not our switches).
    m = min(ppc_branch.shape[0], n_phys)
    r[:m] = np.real(ppc_branch[:m, BR_R]).astype(float)
    x[:m] = np.real(ppc_branch[:m, BR_X]).astype(float)
    b[:m] = np.real(ppc_branch[:m, BR_B]).astype(float)
    ppc_tap = np.real(ppc_branch[:m, BR_TAP]).astype(float)
    ppc_tap[ppc_tap == 0] = 1.0
    tap[:m] = ppc_tap
    shift_deg[:m] = np.real(ppc_branch[:m, BR_SHIFT]).astype(float)
    rate_a[:m] = np.real(ppc_branch[:m, BR_RATE_A]).astype(float)
    br_status[:m] = np.real(ppc_branch[:m, BR_STATUS]).astype(np.int64)
    if ppc_branch.shape[1] > BR_ANGMAX:
        ang_min[:m] = np.real(ppc_branch[:m, BR_ANGMIN]).astype(float)
        ang_max[:m] = np.real(ppc_branch[:m, BR_ANGMAX]).astype(float)

    shift_rad = np.deg2rad(shift_deg)
    Yff, Yft, Ytf, Ytt = compute_branch_admittances(r, x, b, tap, shift_rad)

    return {
        "idx": np.arange(n_branch, dtype=np.int64),
        "from_bus": from_bus,
        "to_bus": to_bus,
        "r": r,
        "x": x,
        "b": b,
        "Yff_r": np.real(Yff),
        "Yff_i": np.imag(Yff),
        "Yft_r": np.real(Yft),
        "Yft_i": np.imag(Yft),
        "Ytf_r": np.real(Ytf),
        "Ytf_i": np.imag(Ytf),
        "Ytt_r": np.real(Ytt),
        "Ytt_i": np.imag(Ytt),
        "tap": tap,
        "shift": shift_deg,
        "ang_min": ang_min,
        "ang_max": ang_max,
        "rate_a": rate_a,
        "br_status": br_status,
        "_n_line": n_line,
        "_n_trafo": n_trafo,
        "_n_switch": n_switch,
        "_switch_idx": np.asarray(switches.index, dtype=np.int64),
    }


def branch_flows(net):
    """Return pf, qf, pt, qt arrays (MW/MVAr) for lines, trafos, then bus-bus switches.

    Missing/NaN results (e.g. de-energised branch) map to 0.0. Switch flows are
    read from ``net.res_switch`` when this pandapower version reports them
    (``p_from_mw`` ...); otherwise they are 0.
    """
    n_line = len(net.line)
    n_trafo = len(net.trafo)
    switches = bus_bus_switches(net)
    n_switch = len(switches)
    n = n_line + n_trafo + n_switch
    pf = np.zeros(n)
    qf = np.zeros(n)
    pt = np.zeros(n)
    qt = np.zeros(n)

    if n_line:
        rl = net.res_line
        pf[:n_line] = np.nan_to_num(
            rl.p_from_mw.reindex(net.line.index).values,
            nan=0.0,
        )
        qf[:n_line] = np.nan_to_num(
            rl.q_from_mvar.reindex(net.line.index).values,
            nan=0.0,
        )
        pt[:n_line] = np.nan_to_num(rl.p_to_mw.reindex(net.line.index).values, nan=0.0)
        qt[:n_line] = np.nan_to_num(
            rl.q_to_mvar.reindex(net.line.index).values,
            nan=0.0,
        )
    if n_trafo:
        rt = net.res_trafo
        pf[n_line : n_line + n_trafo] = np.nan_to_num(
            rt.p_hv_mw.reindex(net.trafo.index).values,
            nan=0.0,
        )
        qf[n_line : n_line + n_trafo] = np.nan_to_num(
            rt.q_hv_mvar.reindex(net.trafo.index).values,
            nan=0.0,
        )
        pt[n_line : n_line + n_trafo] = np.nan_to_num(
            rt.p_lv_mw.reindex(net.trafo.index).values,
            nan=0.0,
        )
        qt[n_line : n_line + n_trafo] = np.nan_to_num(
            rt.q_lv_mvar.reindex(net.trafo.index).values,
            nan=0.0,
        )
    rs = getattr(net, "res_switch", None)
    if n_switch and rs is not None and len(rs):
        s0 = n_line + n_trafo
        for out, col in (
            (pf, "p_from_mw"),
            (qf, "q_from_mvar"),
            (pt, "p_to_mw"),
            (qt, "q_to_mvar"),
        ):
            if col in rs.columns:
                out[s0:] = np.nan_to_num(
                    rs[col].reindex(switches.index).values.astype(float),
                    nan=0.0,
                )
    return pf, qf, pt, qt


# --- VLD extension columns (branch) ---------------------------------------
def branch_col_BRANCH_BASE_STATUS(
    net,
    static,
    base_line_status,
    base_trafo_status,
    base_switch_status=None,
):
    """Healthy-in-base indicator per branch (VLD context).

    For lines / trafos this should be the base *electrical* state, i.e.
    in service AND every line / trafo switch on it closed; for bus-bus switches
    it is closed AND in service in the base network.
    """
    n_line = static["_n_line"]
    n_phys = n_line + static["_n_trafo"]
    base_switch_status = base_switch_status or {}
    out = np.ones(len(static["idx"]), dtype=np.int64)
    for i, line_idx in enumerate(net.line.index):
        out[i] = int(base_line_status.get(int(line_idx), 1))
    for j, trafo_idx in enumerate(net.trafo.index):
        out[n_line + j] = int(base_trafo_status.get(int(trafo_idx), 1))
    for k, sw_idx in enumerate(static.get("_switch_idx", [])):
        out[n_phys + k] = int(base_switch_status.get(int(sw_idx), 1))
    return out


def branch_col_BRANCH_CONTINGENCY(
    net,
    static,
    contingency_lines,
    contingency_trafos,
    contingency_switches=None,
):
    """1 where the branch is a directly-failed contingency element (VLD).

    ``contingency_lines`` / ``contingency_trafos`` should already include the
    lines / trafos whose own line / trafo switch is opened by the scenario;
    ``contingency_switches`` holds the bus-bus switches opened by the scenario.
    """
    n_line = static["_n_line"]
    n_phys = n_line + static["_n_trafo"]
    contingency_switches = contingency_switches or set()
    out = np.zeros(len(static["idx"]), dtype=np.int64)
    for i, line_idx in enumerate(net.line.index):
        out[i] = int(int(line_idx) in contingency_lines)
    for j, trafo_idx in enumerate(net.trafo.index):
        out[n_line + j] = int(int(trafo_idx) in contingency_trafos)
    for k, sw_idx in enumerate(static.get("_switch_idx", [])):
        out[n_phys + k] = int(int(sw_idx) in contingency_switches)
    return out


def branch_col_BRANCH_STATUS(
    net,
    static,
    reachable_buses,
    line_active_fn,
    trafo_active_fn,
    switch_active_fn=None,
):
    """Post-contingency branch energisation label (VLD target).

    1 only when the branch is electrically active AND both endpoints are
    energised in the REF-connected healthy graph.
    """
    n_line = static["_n_line"]
    out = np.zeros(len(static["idx"]), dtype=np.int64)
    fb, tb = static["from_bus"], static["to_bus"]
    for i, line_idx in enumerate(net.line.index):
        active = line_active_fn(net, line_idx)
        out[i] = int(
            active
            and bool(reachable_buses.get(int(fb[i]), False))
            and bool(reachable_buses.get(int(tb[i]), False)),
        )
    for j, trafo_idx in enumerate(net.trafo.index):
        k = n_line + j
        active = trafo_active_fn(net, trafo_idx)
        out[k] = int(
            active
            and bool(reachable_buses.get(int(fb[k]), False))
            and bool(reachable_buses.get(int(tb[k]), False)),
        )
    switch_active_fn = switch_active_fn or switch_is_closed_and_in_service
    n_phys = n_line + static["_n_trafo"]
    for s, sw_idx in enumerate(static.get("_switch_idx", [])):
        k = n_phys + s
        active = switch_active_fn(net, int(sw_idx))
        out[k] = int(
            active
            and bool(reachable_buses.get(int(fb[k]), False))
            and bool(reachable_buses.get(int(tb[k]), False)),
        )
    return out


# ===========================================================================
# Frame assemblers — call the per-column functions in schema order
# ===========================================================================
BUS_DATAKIT_COLUMNS = [
    "load_scenario_idx",
    "bus",
    "Pd",
    "Qd",
    "Pg",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "vn_kv",
    "min_vm_pu",
    "max_vm_pu",
    "GS",
    "BS",
]
BUS_VLD_COLUMNS = ["BUS_BASE_STATUS", "BUS_CONTINGENCY", "BUS_STATUS"]

GEN_DATAKIT_COLUMNS = [
    "load_scenario_idx",
    "idx",
    "bus",
    "p_mw",
    "q_mvar",
    "min_p_mw",
    "max_p_mw",
    "min_q_mvar",
    "max_q_mvar",
    "cp0_eur",
    "cp1_eur_per_mw",
    "cp2_eur_per_mw2",
    "in_service",
    "is_slack_gen",
]

BRANCH_DATAKIT_COLUMNS = [
    "load_scenario_idx",
    "idx",
    "from_bus",
    "to_bus",
    "pf",
    "qf",
    "pt",
    "qt",
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
    "tap",
    "shift",
    "ang_min",
    "ang_max",
    "rate_a",
    "br_status",
]
BRANCH_VLD_COLUMNS = ["BRANCH_BASE_STATUS", "BRANCH_CONTINGENCY", "BRANCH_STATUS"]


def extract_bus_frame(
    net,
    scenario_id,
    load_scenario_idx,
    base_mva,
    reachable_buses,
    base_bus_status,
    contingency_buses,
    include_vld=True,
):
    """Build the bus DataFrame for one solved scenario (datakit + VLD cols)."""
    n = len(net.bus)
    frame = {
        "scenario": np.full(n, scenario_id, dtype=np.int64),
        "load_scenario_idx": np.full(n, load_scenario_idx, dtype=np.int64),
        "bus": bus_index(net),
        "Pd": bus_col_Pd(net),
        "Qd": bus_col_Qd(net),
        "Pg": bus_col_Pg(net),
        "Qg": bus_col_Qg(net),
        "Vm": bus_col_Vm(net),
        "Va": bus_col_Va(net),
        "PQ": bus_col_PQ(net),
        "PV": bus_col_PV(net),
        "REF": bus_col_REF(net),
        "vn_kv": bus_col_vn_kv(net),
        "min_vm_pu": bus_col_min_vm_pu(net),
        "max_vm_pu": bus_col_max_vm_pu(net),
        "GS": bus_col_GS(net, base_mva),
        "BS": bus_col_BS(net, base_mva),
    }
    if include_vld:
        frame["BUS_BASE_STATUS"] = bus_col_BUS_BASE_STATUS(net, base_bus_status)
        frame["BUS_CONTINGENCY"] = bus_col_BUS_CONTINGENCY(net, contingency_buses)
        frame["BUS_STATUS"] = bus_col_BUS_STATUS(net, reachable_buses)
    cols = ["scenario"] + BUS_DATAKIT_COLUMNS + (BUS_VLD_COLUMNS if include_vld else [])
    return pd.DataFrame(frame)[cols]


def extract_gen_frame(net, scenario_id, load_scenario_idx):
    """Build the generator DataFrame for one solved scenario (datakit cols)."""
    n = len(net.gen)
    cp0, cp1, cp2 = _gen_cost_matrix(net)
    frame = {
        "scenario": np.full(n, scenario_id, dtype=np.int64),
        "load_scenario_idx": np.full(n, load_scenario_idx, dtype=np.int64),
        "idx": gen_col_idx(net),
        "bus": gen_col_bus(net),
        "p_mw": gen_col_p_mw(net),
        "q_mvar": gen_col_q_mvar(net),
        "min_p_mw": gen_col_min_p_mw(net),
        "max_p_mw": gen_col_max_p_mw(net),
        "min_q_mvar": gen_col_min_q_mvar(net),
        "max_q_mvar": gen_col_max_q_mvar(net),
        "cp0_eur": cp0,
        "cp1_eur_per_mw": cp1,
        "cp2_eur_per_mw2": cp2,
        "in_service": gen_col_in_service(net),
        "is_slack_gen": gen_col_is_slack_gen(net),
    }
    cols = ["scenario"] + GEN_DATAKIT_COLUMNS
    if (
        n == 0
    ):  # Return an empty dataframe with all the columns of the gen dataset if there is no generator in the scenario
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(frame)[cols]


def extract_branch_frame(
    net,
    scenario_id,
    load_scenario_idx,
    base_mva,
    reachable_buses,
    base_line_status,
    base_trafo_status,
    contingency_lines,
    contingency_trafos,
    line_active_fn,
    trafo_active_fn,
    include_vld=True,
    base_switch_status=None,
    contingency_switches=None,
    switch_active_fn=None,
):
    """Build the branch DataFrame for one solved scenario (datakit + VLD cols).

    One row per line, trafo and bus-bus switch (see ``build_branch_static``).
    """
    static = build_branch_static(net, base_mva)
    pf, qf, pt, qt = branch_flows(net)
    n = len(static["idx"])
    frame = {
        "scenario": np.full(n, scenario_id, dtype=np.int64),
        "load_scenario_idx": np.full(n, load_scenario_idx, dtype=np.int64),
        "idx": static["idx"],
        "from_bus": static["from_bus"],
        "to_bus": static["to_bus"],
        "pf": pf,
        "qf": qf,
        "pt": pt,
        "qt": qt,
        "r": static["r"],
        "x": static["x"],
        "b": static["b"],
        "Yff_r": static["Yff_r"],
        "Yff_i": static["Yff_i"],
        "Yft_r": static["Yft_r"],
        "Yft_i": static["Yft_i"],
        "Ytf_r": static["Ytf_r"],
        "Ytf_i": static["Ytf_i"],
        "Ytt_r": static["Ytt_r"],
        "Ytt_i": static["Ytt_i"],
        "tap": static["tap"],
        "shift": static["shift"],
        "ang_min": static["ang_min"],
        "ang_max": static["ang_max"],
        "rate_a": static["rate_a"],
        "br_status": static["br_status"],
    }
    if include_vld:
        frame["BRANCH_BASE_STATUS"] = branch_col_BRANCH_BASE_STATUS(
            net,
            static,
            base_line_status,
            base_trafo_status,
            base_switch_status,
        )
        frame["BRANCH_CONTINGENCY"] = branch_col_BRANCH_CONTINGENCY(
            net,
            static,
            contingency_lines,
            contingency_trafos,
            contingency_switches,
        )
        frame["BRANCH_STATUS"] = branch_col_BRANCH_STATUS(
            net,
            static,
            reachable_buses,
            line_active_fn,
            trafo_active_fn,
            switch_active_fn,
        )
    cols = (
        ["scenario"]
        + BRANCH_DATAKIT_COLUMNS
        + (BRANCH_VLD_COLUMNS if include_vld else [])
    )
    return pd.DataFrame(frame)[cols]


# ===========================================================================
# Consistency check: flag-based reachability vs the BUS_STATUS label
# ===========================================================================
def reachability_from_flags(bus_frame, branch_frame, sources="ref"):
    """REF-reachability a model can derive from ONE scenario's input flags.

    Mirrors ``VLDTopologyLoss._build_graph_reachability`` on the graph the
    model is given: edges with ``br_status == 1`` (the VLD transforms drop the
    others), a branch is available when BRANCH_BASE_STATUS = 1 and
    BRANCH_CONTINGENCY = 0, a bus when BUS_BASE_STATUS = 1 and
    BUS_CONTINGENCY = 0, and the seeds are the available REF buses (plus PV
    buses for ``sources="ref_and_gen"``). Returns a bool array aligned with the
    rows of ``bus_frame``.
    """
    bus_ids = bus_frame["bus"].to_numpy()
    pos = {int(b): i for i, b in enumerate(bus_ids)}
    n = len(bus_ids)
    available = (bus_frame["BUS_BASE_STATUS"].to_numpy() > 0.5) & (
        bus_frame["BUS_CONTINGENCY"].to_numpy() < 0.5
    )
    is_source = bus_frame["REF"].to_numpy() > 0.5
    if sources == "ref_and_gen":
        is_source = is_source | (bus_frame["PV"].to_numpy() > 0.5)

    keep = (
        (branch_frame["br_status"].to_numpy() > 0.5)
        & (branch_frame["BRANCH_BASE_STATUS"].to_numpy() > 0.5)
        & (branch_frame["BRANCH_CONTINGENCY"].to_numpy() < 0.5)
    )
    adjacency = [[] for _ in range(n)]
    for f, t in zip(
        branch_frame["from_bus"].to_numpy()[keep],
        branch_frame["to_bus"].to_numpy()[keep],
    ):
        i, j = pos[int(f)], pos[int(t)]
        adjacency[i].append(j)
        adjacency[j].append(i)

    reachable = np.zeros(n, dtype=bool)
    stack = [i for i in range(n) if available[i] and is_source[i]]
    for i in stack:
        reachable[i] = True
    while stack:
        i = stack.pop()
        for j in adjacency[i]:
            if available[j] and not reachable[j]:
                reachable[j] = True
                stack.append(j)
    return reachable


def count_reachability_conflicts(bus_df, branch_df, max_scenarios=2000, seed=0):
    """Compare flag-based reachability with BUS_STATUS on (a sample of) scenarios.

    A conflict is a bus whose BUS_STATUS label differs from what the model can
    derive from its inputs (see ``reachability_from_flags``). With switches
    exported as edges and switch contingencies flagged this must be 0; any
    other value means the labels encode information the model cannot see.
    """
    scenarios = np.sort(bus_df["scenario"].unique())
    if max_scenarios is not None and len(scenarios) > max_scenarios:
        rng = np.random.default_rng(seed)
        scenarios = np.sort(rng.choice(scenarios, size=max_scenarios, replace=False))
    bus_groups = bus_df[bus_df["scenario"].isin(scenarios)].groupby("scenario")
    branch_groups = branch_df[branch_df["scenario"].isin(scenarios)].groupby("scenario")
    n_bus = n_conflict = n_reach0_label1 = n_reach1_label0 = 0
    scenarios_with_conflict = 0
    for scenario, bus_frame in bus_groups:
        reach = reachability_from_flags(bus_frame, branch_groups.get_group(scenario))
        label = bus_frame["BUS_STATUS"].to_numpy() > 0.5
        r0l1 = int((~reach & label).sum())
        r1l0 = int((reach & ~label).sum())
        n_bus += len(label)
        n_reach0_label1 += r0l1
        n_reach1_label0 += r1l0
        n_conflict += r0l1 + r1l0
        scenarios_with_conflict += int(r0l1 + r1l0 > 0)
    return {
        "checked_scenarios": int(len(scenarios)),
        "checked_buses": int(n_bus),
        "conflict_buses": int(n_conflict),
        "conflict_rate": float(n_conflict / n_bus) if n_bus else 0.0,
        "conflict_reach0_label1": int(n_reach0_label1),
        "conflict_reach1_label0": int(n_reach1_label0),
        "scenarios_with_conflict": int(scenarios_with_conflict),
    }


def _sample_scenarios(bus_df, max_scenarios, seed):
    scenarios = np.sort(bus_df["scenario"].unique())
    if max_scenarios is not None and len(scenarios) > max_scenarios:
        rng = np.random.default_rng(seed)
        scenarios = np.sort(rng.choice(scenarios, size=max_scenarios, replace=False))
    return scenarios


def count_unexplained_branch_outages(bus_df, branch_df, max_scenarios=2000, seed=0):
    """Branches switched off (br_status = 0) for no reason the model can see.

    ``br_status`` decides which edges the model gets (the VLD transforms drop
    br_status = 0). A branch may legitimately be off because it is off in the
    base case (BRANCH_BASE_STATUS = 0), is a contingency (BRANCH_CONTINGENCY = 1)
    or touches an unavailable bus (BUS_BASE_STATUS = 0 or BUS_CONTINGENCY = 1).
    Any other br_status = 0 - e.g. a solver switching off branches inside a
    de-energised area - would encode the label in the graph itself; this must
    be 0.
    """
    scenarios = _sample_scenarios(bus_df, max_scenarios, seed)
    bus = bus_df.loc[bus_df["scenario"].isin(scenarios), ["scenario", "bus"]].copy()
    bus["available"] = (
        (bus_df.loc[bus.index, "BUS_BASE_STATUS"] > 0.5)
        & (bus_df.loc[bus.index, "BUS_CONTINGENCY"] < 0.5)
    ).to_numpy()
    br = branch_df.loc[
        branch_df["scenario"].isin(scenarios),
        ["scenario", "idx", "from_bus", "to_bus", "br_status", "BRANCH_BASE_STATUS", "BRANCH_CONTINGENCY"],
    ]
    br = br.merge(
        bus.rename(columns={"bus": "from_bus", "available": "from_ok"}),
        on=["scenario", "from_bus"],
        how="left",
    ).merge(
        bus.rename(columns={"bus": "to_bus", "available": "to_ok"}),
        on=["scenario", "to_bus"],
        how="left",
    )
    unexplained = (
        (br["br_status"] < 0.5)
        & (br["BRANCH_BASE_STATUS"] > 0.5)
        & (br["BRANCH_CONTINGENCY"] < 0.5)
        & br["from_ok"].fillna(False).astype(bool)
        & br["to_ok"].fillna(False).astype(bool)
    )
    bad = br.loc[unexplained, ["scenario", "idx"]]
    return {
        "checked_scenarios": int(len(scenarios)),
        "checked_branches": int(len(br)),
        "unexplained_off_branches": int(unexplained.sum()),
        "scenarios_with_unexplained": int(bad["scenario"].nunique()),
        "examples": bad.head(5).to_dict("records"),
    }
