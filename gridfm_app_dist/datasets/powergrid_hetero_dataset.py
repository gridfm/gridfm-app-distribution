from gridfm_app_dist.datasets.normalizers import Normalizer

import os.path as osp
import os
import torch
from torch_geometric.data import Dataset
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable
from torch_geometric.data import HeteroData
from gridfm_app_dist.datasets.globals import PG_H  # ,VA_H


#####################
def _optional_cols(df, cols, default_value):
    """
    Return a list of column names guaranteed to exist in df.

    Matching is CASE-INSENSITIVE: if a requested name (e.g. ``bus_status_target``)
    is not present but a column with the same name up to case (e.g. the parquet's
    ``BUS_STATUS``) exists, the real column name is returned so downstream reads
    hit the actual data instead of silently falling back to ``default_value``.
    Only when no case-insensitive match exists is the column created and filled
    with the corresponding default.
    """
    if len(default_value) < len(cols):
        raise ValueError(
            f"default_value has length {len(default_value)} but cols has length {len(cols)}",
        )
    lower_map = {c.lower(): c for c in df.columns}
    resolved = []
    for i, col in enumerate(cols):
        real = lower_map.get(col.lower())
        if real is None:
            df[col] = default_value[i]
            real = col
        resolved.append(real)
    return resolved


#####################


class HeteroGridDatasetDisk(Dataset):
    """
    A PyTorch Geometric `Dataset` for power grid data stored on disk.
    This dataset reads node and edge CSV files and saves each graph
    separately on disk as a processed file. Data is loaded from disk
    lazily on demand. Normalization is applied at access time via
    the data_normalizer (which must be fitted externally before iteration).

    Args:
        root (str): Root directory where the dataset is stored.
        data_normalizer (Normalizer): Normalizer used for features (fitted externally by the datamodule).
        transform (callable, optional): Transformation applied at runtime.
        pre_transform (callable, optional): Transformation applied before saving to disk.
        pre_filter (callable, optional): Filter to determine which graphs to keep.
    """

    def __init__(
        self,
        root: str,
        data_normalizer: Normalizer,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.data_normalizer = data_normalizer
        self.length = None
        self.has_generators = None

        super().__init__(root, transform, pre_transform, pre_filter)

        load_scenarios_path = osp.join(self.processed_dir, "load_scenarios.pt")
        if osp.exists(load_scenarios_path):
            self.load_scenarios = torch.load(load_scenarios_path, weights_only=True)

        has_gen_path = osp.join(self.processed_dir, "has_generators.pt")
        if osp.exists(has_gen_path):
            self.has_generators = bool(
                torch.load(has_gen_path, weights_only=True).item(),
            )

    @property
    def raw_file_names(self):
        return ["bus_data.parquet", "gen_data.parquet", "branch_data.parquet"]

    @property
    def processed_done_file(self):
        return "processed_raw_files.done"

    @property
    def processed_file_names(self):
        return ["load_scenarios.pt", "has_generators.pt", self.processed_done_file]

    def download(self):
        pass

    def process(self):
        print("LOADING DATA")
        bus_data = pd.read_parquet(osp.join(self.raw_dir, "bus_data.parquet"))
        branch_data = pd.read_parquet(osp.join(self.raw_dir, "branch_data.parquet"))

        #####################
        # Generator data is OPTIONAL. A network may legitimately have no generators, in which case gen_data.parquet is either missing or present-but-empty.
        # We load it defensively and flag `has_generators` so the rest of process() can build a valid *empty* ``gen`` node set instead of crashing.
        # Tasks/losses/model are unchanged: an empty gen store is natively supported by PyG HeteroData.
        gen_path = osp.join(self.raw_dir, "gen_data.parquet")
        if osp.exists(gen_path):
            gen_data = pd.read_parquet(gen_path)
        else:
            print("gen_data.parquet not found -> treating dataset as generator-free.")
            gen_data = pd.DataFrame()
        self.has_generators = not gen_data.empty
        if not self.has_generators:
            print("No generators in dataset -> building empty gen node sets.")
        # Persist the flag so cached runs (which skip process()) can restore it
        # in __init__ without re-reading the raw parquet.
        torch.save(
            torch.tensor(self.has_generators),
            osp.join(self.processed_dir, "has_generators.pt"),
        )
        #####################

        assert (
            bus_data["scenario"].min() == 0
            and bus_data["scenario"].max() == len(bus_data["scenario"].unique()) - 1
        )

        load_scenarios = torch.tensor(
            bus_data.groupby("scenario", sort=True)["load_scenario_idx"].first().values,
        )
        torch.save(load_scenarios, osp.join(self.processed_dir, "load_scenarios.pt"))

        #####################
        # Aggregate per-bus reactive-generation limits only when generators
        # exist. Without generators the bus columns default to 0 (the same
        # value the left-merge + fillna(0) would have produced).
        if self.has_generators:
            agg_gen = (
                gen_data.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]]
                .sum()
                .reset_index()
            )
            bus_data = bus_data.merge(
                agg_gen, on=["scenario", "bus"], how="left",
            ).fillna(0)
        else:
            bus_data["min_q_mvar"] = 0.0
            bus_data["max_q_mvar"] = 0.0
        #####################

        done_path = osp.join(self.processed_dir, self.processed_done_file)
        if osp.exists(done_path):
            print("Processed files already exist. Skipping processing.")
            return

        bus_features = [
            "Pd",
            "Qd",
            "Qg",
            "Vm",
            "Va",
            "PQ",
            "PV",
            "REF",
            "min_vm_pu",
            "max_vm_pu",
            "min_q_mvar",
            "max_q_mvar",
            "GS",
            "BS",
            "vn_kv",
        ]

        #####################
        # Optional VLD bus input columns
        vld_bus_input_features = _optional_cols(
            bus_data, ["bus_base_status", "bus_contingency"], default_value=[1.0, 0.0],
        )
        bus_features = bus_features + vld_bus_input_features
        #####################

        gen_features = [
            "p_mw",
            "min_p_mw",
            "max_p_mw",
            "cp0_eur",
            "cp1_eur_per_mw",
            "cp2_eur_per_mw2",
            "in_service",
        ]
        common_branch_features = ["tap", "ang_min", "ang_max", "rate_a", "br_status"]

        #####################
        # Optional VLD branch topology columns
        vld_branch_features = _optional_cols(
            branch_data,
            ["branch_base_status", "branch_contingency"],
            default_value=[1.0, 0.0],
        )
        common_branch_features = common_branch_features + vld_branch_features
        #####################

        forward_branch_features = [
            "pf",
            "qf",
            "Yff_r",
            "Yff_i",
            "Yft_r",
            "Yft_i",
        ] + common_branch_features
        reverse_branch_features = [
            "pt",
            "qt",
            "Ytt_r",
            "Ytt_i",
            "Ytf_r",
            "Ytf_i",
        ] + common_branch_features

        #####################
        # Optional VLD target column on bus.y.
        vld_bus_target_features = _optional_cols(
            bus_data, ["BUS_STATUS", "bus_status_target"], default_value=[1.0, 1.0],
        )[:1]
        #####################

        #####################
        # Number of generator feature columns, used to shape the empty gen
        # tensors when the dataset has no generators.
        n_gen_features = len(gen_features)
        #####################

        # Group by scenario
        bus_groups = bus_data.groupby("scenario")
        gen_groups = gen_data.groupby("scenario") if self.has_generators else None
        branch_groups = branch_data.groupby("scenario")

        # Process each scenario
        for scenario in tqdm(
            bus_data["scenario"].unique(), desc="Processing scenarios",
        ):
            if osp.exists(osp.join(self.processed_dir, f"data_index_{scenario}.pt")):
                continue
            #####################
            # Branch data is always required. Generator data is optional: only
            # require a gen group for this scenario when the dataset actually
            # has generators.
            if scenario not in branch_groups.groups:
                raise ValueError(f"Scenario {scenario} has no branch data.")
            if self.has_generators and scenario not in gen_groups.groups:
                raise ValueError(f"Scenario {scenario} has no generator data.")
            #####################

            data = HeteroData()

            # Bus nodes
            bus_df = bus_groups.get_group(scenario)
            data["bus"].x = torch.tensor(bus_df[bus_features].values, dtype=torch.float)

            #####################
            # Bus ids in the raw parquet are the network's ORIGINAL labels and
            # are NOT guaranteed to be contiguous 0..N-1 (e.g. they may range
            # 10..142 for a 44-bus network). bus.x is built positionally, so
            # node i corresponds to the i-th row of bus_df. Every edge that
            # references a bus (branch from_bus/to_bus, gen<->bus) uses these
            # original labels and must be remapped to the compact node position
            # 0..N-1, otherwise edge_index points out of bounds and the model's
            # message passing gather raises a CUDA device-side assert.
            #
            # Build the {original_bus_id -> node_position} map from bus_df's row
            # order, which is the single source of truth for node indexing.
            bus_id_to_pos = {
                int(orig_id): pos for pos, orig_id in enumerate(bus_df["bus"].values)
            }

            def _map_bus_ids(series, ctx):
                """Vectorized remap of a pandas bus-id Series to node positions.
                Raises a clear, scenario-tagged error if an id is unknown."""
                mapped = series.map(bus_id_to_pos)
                if mapped.isna().any():
                    missing = sorted(set(series[mapped.isna()].tolist()))
                    raise ValueError(
                        f"Scenario {scenario}: {ctx} references bus id(s) "
                        f"{missing} not present in bus_data (valid ids: "
                        f"{sorted(bus_id_to_pos)[:5]}... n={len(bus_id_to_pos)}).",
                    )
                return torch.tensor(mapped.astype("int64").values, dtype=torch.long)

            #####################

            # Generator nodes (may be empty if the network has no generators)
            #####################
            if self.has_generators:
                gen_df = gen_groups.get_group(scenario).reset_index()
                data["gen"].x = torch.tensor(
                    gen_df[gen_features].values, dtype=torch.float,
                )
                gen_df["gen_index"] = gen_df.index  # Use actual index as generator ID
            else:
                gen_df = None
                # Empty gen node feature matrix with the correct feature width.
                data["gen"].x = torch.zeros((0, n_gen_features), dtype=torch.float)
            #####################

            #####################
            # Additional columns for the VLD task
            bus_y_base = torch.tensor(
                bus_df[["Pd", "Qd", "Qg", "Vm", "Va"]].values, dtype=torch.float,
            )  # Base label columns: [Pd, Qd, Qg, Vm, Va]  (indices PD_H..VA_H = 0..4)
            bus_y_vld = torch.tensor(
                bus_df[vld_bus_target_features].values, dtype=torch.float,
            )  # Optional VLD target column: bus_status_target  (index BUS_STATUS_TARGET = 5)
            bus_y_pg = torch.tensor(
                bus_df[["Pg"]].values, dtype=torch.float,
            )  # Net per-bus active generation Pg (gens + ext_grid), APPENDED as the last label column (index PG_BUS_H = 6).
            data["bus"].y = torch.cat([bus_y_base, bus_y_vld, bus_y_pg], dim=1)
            #####################

            data["gen"].y = (
                data["gen"].x[:, : (PG_H + 1)].clone()
            )  # [0, PG_H+1] when empty

            # Bus-Bus edges
            branch_df = branch_groups.get_group(scenario)
            #####################
            # Remap branch endpoint ids from original bus labels to node
            # positions before using them as edge indices.
            from_pos = _map_bus_ids(branch_df["from_bus"], "branch from_bus")
            to_pos = _map_bus_ids(branch_df["to_bus"], "branch to_bus")
            forward_edges = torch.stack([from_pos, to_pos])
            #####################
            forward_edge_attr = torch.tensor(
                branch_df[forward_branch_features].values, dtype=torch.float,
            )
            reverse_edges = torch.stack([to_pos, from_pos])
            reverse_edge_attr = torch.tensor(
                branch_df[reverse_branch_features].values, dtype=torch.float,
            )

            edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
            edge_attr = torch.cat([forward_edge_attr, reverse_edge_attr], dim=0)

            forward_targets = torch.tensor(
                branch_df[["pf", "qf"]].values, dtype=torch.float,
            )
            reverse_targets = torch.tensor(
                branch_df[["pt", "qt"]].values, dtype=torch.float,
            )
            edge_y = torch.cat([forward_targets, reverse_targets], dim=0)

            data["bus", "connects", "bus"].edge_index = edge_index
            data["bus", "connects", "bus"].edge_attr = edge_attr
            data["bus", "connects", "bus"].y = edge_y

            # Gen-Bus and Bus-Gen edges (empty [2, 0] when there are no gens)
            #####################
            if self.has_generators:
                # Remap the gen 'bus' column (original bus ids) to node positions; gen_index is already 0..Ng-1 from reset_index().
                gen_bus_pos = _map_bus_ids(gen_df["bus"], "gen bus")
                gen_idx = torch.tensor(
                    gen_df["gen_index"].astype("int64").values, dtype=torch.long,
                )
                data["gen", "connected_to", "bus"].edge_index = torch.stack(
                    [gen_idx, gen_bus_pos],
                )
                data["bus", "connected_to", "gen"].edge_index = torch.stack(
                    [gen_bus_pos, gen_idx],
                )
            else:
                data["gen", "connected_to", "bus"].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long,
                )
                data["bus", "connected_to", "gen"].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long,
                )
            #####################
            data["scenario_id"] = torch.tensor([scenario], dtype=torch.long)

            # Save graph
            torch.save(
                data.to_dict(),
                osp.join(self.processed_dir, f"data_index_{scenario}.pt"),
            )

        with open(osp.join(self.processed_dir, self.processed_done_file), "w") as f:
            f.write("done")

    def len(self):
        if self.length is None:
            files = [
                f
                for f in os.listdir(self.processed_dir)
                if f.startswith("data_index_") and f.endswith(".pt")
            ]
            self.length = len(files)
        return self.length

    def get(self, idx):
        file_name = osp.join(self.processed_dir, f"data_index_{idx}.pt")
        if not osp.exists(file_name):
            raise IndexError(f"Data file {file_name} does not exist.")
        data_dict = torch.load(file_name, weights_only=True)
        data = HeteroData.from_dict(data_dict)
        self.data_normalizer.transform(data=data)
        return data
