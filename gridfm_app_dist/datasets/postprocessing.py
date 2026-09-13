import numpy as np
from scipy.sparse import csr_matrix


def compute_branch_currents_kA(Yf, Yt, V, Vf_base_kV, Vt_base_kV, sn_mva):
    """
    TODO docstrings
    """

    If_pu = Yf @ V  # From-end currents in per-unit (I_f = Y_f V)
    If_kA = np.abs(If_pu) * sn_mva / (np.sqrt(3) * Vf_base_kV)  # Conversion to kA

    # Construct to-end admittance matrix Yt:
    # Yt[b, :] = y_tf_b * e_f + y_tt_b * e_t
    It_pu = Yt @ V  # To-end currents in per-unit (I_t = Y_t V)
    It_kA = np.abs(It_pu) * sn_mva / (np.sqrt(3) * Vt_base_kV)  # Conversion to kA

    return If_kA, It_kA


def compute_loading(If_kA, It_kA, Vf_base_kV, Vt_base_kV, rate_a):
    """
    Compute per-branch loading using current magnitudes and branch ratings.

    Parameters:
    - edge_index: np.ndarray of shape (n_edges, 2), each row is [from_bus, to_bus]
    - If_kA: np.ndarray of from-side current magnitudes in kA
    - It_kA: np.ndarray of to-side current magnitudes in kA
    - base_kv: np.ndarray of shape (n_buses,), base voltage in kV per bus
    - edge_attr: np.ndarray of shape (n_edges, >=5), edge features, column 4 = RATE_A

    Returns:
    - loading: np.ndarray of shape (n_edges,), max of from and to side loading
    """

    limitf = rate_a / (Vf_base_kV * np.sqrt(3))
    limitt = rate_a / (Vt_base_kV * np.sqrt(3))

    loadingf = If_kA / limitf
    loadingt = It_kA / limitt

    return np.maximum(loadingf, loadingt)


def create_admittance_matrix(bus_params, edge_params, sn_mva=100):
    """
    TODO Docstrings

    Parameters:
    - bus_params: pandas df
    - edge_params: pandas df

    """

    base_kv = bus_params["baseKV"].values

    #####################
    # Bus ids in the raw tables are the network's ORIGINAL labels and are NOT
    # guaranteed to be contiguous 0..nb-1 (e.g. they may range 10..142 for a
    # 44-bus network). base_kv and the sparse admittance columns below are
    # indexed POSITIONALLY (row i of bus_params == column i), so from_bus/to_bus
    # must be remapped to node positions first, exactly as in the dataset
    # process(). If bus_params carries a 'bus' id column we map against it;
    # otherwise the ids are assumed already positional and the map is the
    # identity (this branch is a no-op for contiguous 0..nb-1 data).
    if "bus" in bus_params.columns:
        bus_id_to_pos = {
            int(orig_id): pos for pos, orig_id in enumerate(bus_params["bus"].values)
        }

        def _map_bus_ids(series, ctx):
            mapped = series.map(bus_id_to_pos)
            if mapped.isna().any():
                missing = sorted(set(series[mapped.isna()].tolist()))
                raise ValueError(
                    f"create_admittance_matrix: {ctx} references bus id(s) "
                    f"{missing} not present in bus_params "
                    f"(n_bus={len(bus_id_to_pos)}).",
                )
            return mapped.astype(np.int32).values

        f = _map_bus_ids(edge_params["from_bus"], "from_bus")
        t = _map_bus_ids(edge_params["to_bus"], "to_bus")
    else:
        # No explicit id column: rows are already positional (0..nb-1).
        f = edge_params["from_bus"].values.astype(np.int32)
        t = edge_params["to_bus"].values.astype(np.int32)
    #####################

    # Extract branch admittance coefficients
    Yff = edge_params["Yff_r"].values + 1j * edge_params["Yff_i"].values
    Yft = edge_params["Yft_r"].values + 1j * edge_params["Yft_i"].values
    Ytf = edge_params["Ytf_r"].values + 1j * edge_params["Ytf_i"].values
    Ytt = edge_params["Ytt_r"].values + 1j * edge_params["Ytt_i"].values

    # Get base voltages for the from and to buses (for kA conversion)
    Vf_base_kV = base_kv[f]
    Vt_base_kV = base_kv[t]

    nl = edge_params.shape[0]
    nb = bus_params.shape[0]

    # i = [0, 1, ..., nl-1, 0, 1, ..., nl-1], used for constructing Yf and Yt
    i = np.hstack([np.arange(nl), np.arange(nl)])

    # Construct from-end admittance matrix Yf using the linear combination:
    # Yf[b, :] = y_ff_b * e_f + y_ft_b * e_t
    Yf = csr_matrix((np.hstack([Yff, Yft]), (i, np.hstack([f, t]))), shape=(nl, nb))
    Yt = csr_matrix((np.hstack([Ytf, Ytt]), (i, np.hstack([f, t]))), shape=(nl, nb))

    return Yf, Yt, Vf_base_kV, Vt_base_kV
