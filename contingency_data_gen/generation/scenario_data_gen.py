"""
scenario_data_gen.py
=================

Main script for running the pipeline for generating network data for GridFM training
Automatically loads the network and parameters from the yaml file and source network file
Output datasets contain the graph features for all contingency scenarios

Format of output datasets:
    <raw_dir>/bus_data.parquet/scenario_partition=<k>/*.parquet
    <raw_dir>/gen_data.parquet/scenario_partition=<k>/*.parquet
    <raw_dir>/branch_data.parquet/scenario_partition=<k>/*.parquet

For more details, read the documentation in the read me file
Custom libraries required to generate contigency scenarios
panda_network_builder
datakit_extractors
datakit_writer
excel_pandapower
excel_pandapower_2
excel_pandapower_3
"""

import pandas as pd
import pandapower as pp
from tqdm import tqdm
import itertools
import math
from typing import List, Tuple, Dict, Generator
from typing import Optional
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
import contextlib

# gridfm-datakit-format extraction + partitioned-parquet writing
from contingency_data_gen.generation import datakit_extractors as dke
from contingency_data_gen.generation import datakit_writer as dkw
from contingency_data_gen.generation.excel_pandapower import (  # noqa: F401
    create_network_from_excel,
    create_pandapower_network,
    load_excel_data_to_dicts,
    run_load_flow_and_display_results,
)
from contingency_data_gen.generation.excel_pandapower_2 import ExcelToPandapowerImporter
from contingency_data_gen.generation.excel_pandapower_3 import (
    ExcelToPandapowerImporter2,
)

# Global variable inside worker processes
_worker_generator = None


# Load network once per worker process
def worker_initializer(network_file_path, flag, panda_net, output_dir, name):
    global _worker_generator
    _worker_generator = ScenarioGenerator(
        network_file_path=network_file_path,
        flag=flag,
        net=panda_net,
        output_dir=output_dir,
        prepare_output_dir=False,
        name=name,
    )
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        _worker_generator.load_network_from_excel()  # Suppress output while loading


# Execution process inside each worker. Network already loaded
def process_batch_worker(n_level, scenarios_batch, batch_count, scenario_start_idx):
    # global _worker_generator
    bus_results, gen_results, branch_results = (
        _worker_generator.run_batch_contingency_analysis(
            n_level,
            scenarios_batch,
            scenario_start_idx,
        )
    )
    _worker_generator.save_batch_results(
        bus_results,
        gen_results,
        branch_results,
        batch_count,
    )
    del bus_results, gen_results, branch_results
    return {
        "n_level": n_level,
        "num_scenarios": len(scenarios_batch),
        "batch_index": batch_count,
        "scenario_start_idx": scenario_start_idx,
    }


# Create a new directory and delete existing one if any
def create_dir(path) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


class ScenarioGenerator:
    """
    Memory-optimized contingency generator for N-1, N-2, and N-3 scenarios with sequential batched CSV output in single folder
    """

    def __init__(
        self,
        network_file_path: str,
        flag: Optional[int],
        net: Optional[dict],
        output_dir: str,
        name,
        prepare_output_dir=True,
    ):
        """
        Initialize the contingency generator
        Parameters:
        -----------
        network_file_path : str - Path to Excel file with network data
        pandapower network : dict -Pandapower network imported as a dictionary
        flag: int or None - None when the pandapower network is loaded and 1,2,3 (int) when the pandapower network is loaded from an Excel raw file (3 methods)
        output_dir : str - Output directory for results
        """
        self.network_file_path = network_file_path
        self.output_dir = output_dir
        self.component_indices = {}
        self.total_scenarios = 0
        self.total_batches_processed = 0
        self.global_batch_counter = (
            0  # Global counter for all batches across all N-levels
        )
        self.time_taken = 0
        self.flag = flag
        self.panda_net = net
        self.net = None
        self.n_level = None
        self.raw_dir = None
        self.branch_templates = []
        self.name = name

        # Create output directory
        # Only the main generator should ever clean the directory
        if self.output_dir is not None and prepare_output_dir:
            create_dir(self.output_dir)  # Main output directory

            raw_dir = os.path.join(self.output_dir, "raw")
            create_dir(raw_dir)
            self.raw_dir = raw_dir  # Directory with the raw csv files

            processed_dir = os.path.join(self.output_dir, "processed")
            create_dir(processed_dir)  # Directory with the pt files during training

        elif self.output_dir is not None:
            # Workers: ensure directory exists, but never delete it
            os.makedirs(self.output_dir, exist_ok=True)
            self.raw_dir = os.path.join(self.output_dir, "raw")

    def load_network_from_excel(self):
        """
        Load network from Excel file
        """
        # Load network
        if (
            self.flag is None or self.flag == 9 or self.flag == 10
        ):  # Pandapoawer network loaded in memory
            self.net = self.panda_net
            print("Loading network from pandapower")

        elif self.flag == 1:  # Pandapower network loaded from Excel (Method 1)
            self.net = create_network_from_excel(
                self.network_file_path,
                run_power_flow=False,
            )
            print("Loading network from Excel... using Method 1")

        elif self.flag == 2:  # Pandapower network loaded from Excel (Method 2)
            network = ExcelToPandapowerImporter(self.network_file_path)
            self.net = network.create_pandapower_network()
            print("Loading network from Excel...using Method 2")

        elif self.flag == 3:  # Pandapower network loaded from Excel (Method 3)
            network = ExcelToPandapowerImporter2(self.network_file_path)
            self.net = network.create_pandapower_network()
            print("Loading network from Excel...using Method3")

        if self.net is None:
            raise ValueError(
                "Failed to load network from Excel file if flag is 1 or pandapower network if flag is None",
            )

        # Store component indices
        self._index_components()
        self._build_branch_templates()
        print(
            f"Network loaded: {len(self.net.bus)} buses, {len(self.net.line)} lines, {len(self.net.gen)} generators, {len(self.net.trafo)} trafos, {len(self.net.switch)} switches",
        )

    def _index_components(self):
        """
        Index all components for contingency analysis
        """
        # Get reference bus (skip in contingencies)
        self.ref_bus = (
            self.net.ext_grid.bus.iloc[0] if len(self.net.ext_grid) > 0 else None
        )
        self.component_indices["buses"] = [
            idx for idx in self.net.bus.index if idx != self.ref_bus
        ]  # Buses (excluding reference bus)
        self.component_indices["lines"] = list(self.net.line.index)  # Lines
        self.component_indices["gens"] = list(self.net.gen.index)  # Generators
        self.component_indices["trafos"] = list(self.net.trafo.index)  # Transformers
        self.component_indices["switches"] = list(self.net.switch.index)  # Switches

        # All components as tuples (type, index)
        self.all_components = (
            [("bus", idx) for idx in self.component_indices["buses"]]
            + [("line", idx) for idx in self.component_indices["lines"]]
            + [("gen", idx) for idx in self.component_indices["gens"]]
            + [("trafo", idx) for idx in self.component_indices["trafos"]]
            + [("switch", idx) for idx in self.component_indices["switches"]]
        )

        print(
            f"Indexed {len(self.component_indices['buses'])} buses, "
            f"{len(self.component_indices['lines'])} lines, "
            f"{len(self.component_indices['gens'])} gens, "
            f"{len(self.component_indices['trafos'])} trafos, "
            f"{len(self.component_indices['switches'])} switches, "
            f"{len(self.component_indices['gens'])} generators for contingencies",
        )

    def _build_branch_templates(self):
        """
        Build a stable list of branch templates keyed by bus pair.

        Each template aggregates the physical components that can connect a bus pair.
        This keeps the branch CSV schema stable across scenarios and allows us to export
        both admittance and service state even when a branch is switched off.
        """
        branch_map = {}

        def add_component(bus_i, bus_j, comp_type, comp_idx):
            pair = tuple(sorted((int(bus_i), int(bus_j))))
            entry = branch_map.setdefault(
                pair,
                {"bus_i": pair[0], "bus_j": pair[1], "components": []},
            )
            entry["components"].append((comp_type, int(comp_idx)))

        for line_idx, row in self.net.line.iterrows():
            add_component(row.from_bus, row.to_bus, "line", line_idx)

        for trafo_idx, row in self.net.trafo.iterrows():
            add_component(row.hv_bus, row.lv_bus, "trafo", trafo_idx)

        if len(self.net.switch) > 0:
            bus_bus_switches = self.net.switch[self.net.switch.et == "b"]
            for switch_idx, row in bus_bus_switches.iterrows():
                add_component(row.bus, row.element, "switch_b", switch_idx)

        self.branch_templates = [branch_map[key] for key in sorted(branch_map)]
        print(f"Built {len(self.branch_templates)} branch templates")

    def generate_contingency_scenarios(
        self,
        n_level: int,
        max_size,
    ) -> Generator[List[Tuple], None, None]:
        """
        Generate contingency scenarios for N-k level with batching
        Parameters:
        -----------
        n_level : int - Contingency level (1, 2, or 3)
        batch_size : int (optional) - Number of scenarios per batch. If None, calculates based on memory optimization
        Yields:
        ----------
        list of Batch of scenarios, each scenario is a list of (component_type, component_index) tuples
        """
        if n_level < 1 or n_level > 3:
            raise ValueError("n_level must be 1, 2, or 3")
        self.n_level = n_level
        batch_size = self._calculate_optimal_batch_size(max_size)
        if batch_size is None:
            print("Size of the CSV file is missing")
            return

        components = self.all_components
        total_scenarios = math.comb(len(components), n_level)
        self.total_scenarios = total_scenarios
        print(
            f"Generating {total_scenarios} N-{n_level} contingency scenarios in batches of {batch_size}",
        )

        # Generate all combinations
        scenarios = itertools.combinations(components, n_level)

        current_batch = []
        for i, scenario in enumerate(scenarios):
            current_batch.append(list(scenario))

            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        # Yield remaining scenarios in the last batch
        if current_batch:
            yield current_batch

    def _calculate_optimal_batch_size(self, max_size) -> int:
        """
        Calculate optimal batch size based on network complexity based on the max size of the CSV files
        """
        # Estimate memory usage per scenario
        bus_memory = (
            len(self.net.bus) * 120
        )  # scenario, bus id, node features, BUS_BASE_STATUS, BUS_CONTINGENCY, BUS_STATUS
        branch_memory = (
            len(self.branch_templates) * 72
        )  # scenario, bus pair, G, B, BRANCH_BASE_STATUS, BRANCH_CONTINGENCY, BRANCH_STATUS
        total_per_scenario = bus_memory + branch_memory

        # Target: keep batches under max_size
        if max_size is not None:
            target_memory = max_size * 1024 * 1024  # max_size is in MB
            batch_size = max(1, min(1000, target_memory // total_per_scenario))
        else:
            print("Size of the CSV file is missing")
            return
        print(f"Calculated optimal batch size: {batch_size}")
        return batch_size

    def apply_contingency(self, net_copy, scenario: List[Tuple]):
        """
        Apply contingency scenario to network copy
        Parameters:
        net_copy : pandapower network Copy of the original network
        scenario : list of (component_type, component_index) tuples to disable
        """
        for comp_type, comp_idx in scenario:
            if comp_type == "bus":
                self._disable_bus_components(net_copy, comp_idx)
            elif comp_type == "line":
                net_copy.line.at[comp_idx, "in_service"] = False
            elif comp_type == "gen":
                net_copy.gen.at[comp_idx, "in_service"] = False
            elif comp_type == "trafo":
                net_copy.trafo.at[comp_idx, "in_service"] = False
            elif comp_type == "switch":
                net_copy.switch.at[comp_idx, "closed"] = False

    def _disable_bus_components(self, net, bus_idx):
        """
        Disable all components connected to a bus
        """
        net.bus.at[bus_idx, "in_service"] = False

        # Disable loads at this bus
        load_mask = net.load.bus == bus_idx
        net.load.loc[load_mask, "in_service"] = False

        # Disable generators at this bus
        gen_mask = net.gen.bus == bus_idx
        net.gen.loc[gen_mask, "in_service"] = False

        # Disable lines connected to this bus
        line_mask = (net.line.from_bus == bus_idx) | (net.line.to_bus == bus_idx)
        net.line.loc[line_mask, "in_service"] = False

        # Disable trafos connected to this bus
        trafo_mask = (net.trafo.hv_bus == bus_idx) | (net.trafo.lv_bus == bus_idx)
        net.trafo.loc[trafo_mask, "in_service"] = False

        # Open switches connected to this bus
        switch_mask = net.switch.bus == bus_idx
        if "element" in net.switch.columns and len(net.switch) > 0:
            switch_mask = switch_mask | (
                (net.switch.et == "b") & (net.switch.element == bus_idx)
            )
        net.switch.loc[switch_mask, "closed"] = False

    def run_batch_contingency_analysis(
        self,
        n_level: int,
        scenarios: List[List[Tuple]],
        scenario_start_idx: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run a batch of contingency scenarios in a memory-efficient way.
        - Accumulates per-scenario bus/gen/branch results in lists.
        - Returns three concatenated frames in gridfm-datakit column layout.
        """
        bus_results_list = []
        gen_results_list = []
        branch_results_list = []
        base_scenario_id = scenario_start_idx

        # Cache original statuses once
        original_bus_status = self.net.bus.in_service.to_numpy(copy=True)
        original_line_status = self.net.line.in_service.to_numpy(copy=True)
        original_gen_status = self.net.gen.in_service.to_numpy(copy=True)
        original_trafo_status = self.net.trafo.in_service.to_numpy(copy=True)
        original_switch_status = self.net.switch.closed.to_numpy(copy=True)
        original_load_status = self.net.load.in_service.to_numpy(copy=True)
        base_bus_status = {
            int(bus_idx): int(bool(status))
            for bus_idx, status in zip(self.net.bus.index, original_bus_status)
        }
        # Per-element base health (line / trafo) for the datakit per-branch layout.
        base_line_status = {
            int(line_idx): int(bool(status))
            for line_idx, status in zip(self.net.line.index, original_line_status)
        }
        base_trafo_status = {
            int(trafo_idx): int(bool(status))
            for trafo_idx, status in zip(self.net.trafo.index, original_trafo_status)
        }

        print(f"Running N-{n_level} batch with {len(scenarios)} scenarios...")

        for scenario_idx, scenario in enumerate(
            tqdm(scenarios, desc=f"N-{n_level} Batch"),
            start=0,
        ):
            global_scenario_id = (
                base_scenario_id + scenario_idx
            )  # update global id for this scenario

            # reset state of network
            self.net.bus.in_service.values[:] = original_bus_status
            self.net.line.in_service.values[:] = original_line_status
            self.net.gen.in_service.values[:] = original_gen_status
            self.net.trafo.in_service.values[:] = original_trafo_status
            self.net.switch.closed.values[:] = original_switch_status
            self.net.load.in_service.values[:] = original_load_status

            # Apply contingencies in the scenario
            self.apply_contingency(self.net, scenario)

            # ensure ext_grid buses stay active
            for _, row in self.net.ext_grid.iterrows():
                bus_idx = row.bus
                if not self.net.bus.at[bus_idx, "in_service"]:
                    self.net.bus.at[bus_idx, "in_service"] = True
            try:
                pp.runpp(self.net, numba=True)  # Run power flow
            except Exception as e:
                print(f"Power flow failed for scenario {global_scenario_id}: {e}")
                continue

            if not self.net.converged:
                continue  # skip when the power flow doesn't converge

            try:
                reachable_buses = self._compute_reachable_buses(self.net)
                bus_df = self._extract_bus_results(
                    self.net,
                    global_scenario_id,
                    reachable_buses,
                    scenario,
                    base_bus_status,
                )
                gen_df = self._extract_gen_results(self.net, global_scenario_id)
                branch_df = self._extract_branch_results(
                    self.net,
                    global_scenario_id,
                    reachable_buses,
                    scenario,
                    base_line_status,
                    base_trafo_status,
                )
                if not bus_df.empty and not branch_df.empty:
                    bus_results_list.append(bus_df)
                    # if not gen_df.empty:
                    gen_results_list.append(gen_df)
                    branch_results_list.append(branch_df)
            except Exception as e:
                print(
                    f"Result extraction failed for scenario {global_scenario_id}: {e}",
                )

        # Combine batch results
        bus_results = (
            pd.concat(bus_results_list, ignore_index=True)
            if bus_results_list
            else pd.DataFrame()
        )
        gen_results = (
            pd.concat(gen_results_list, ignore_index=True)
            if gen_results_list
            else pd.DataFrame()
        )
        branch_results = (
            pd.concat(branch_results_list, ignore_index=True)
            if branch_results_list
            else pd.DataFrame()
        )

        return bus_results, gen_results, branch_results

    def _extract_bus_results(
        self,
        net,
        scenario_id: int,
        reachable_buses: Dict[int, bool],
        scenario: List[Tuple],
        base_bus_status: Dict[int, int],
        load_scenario_idx: int = 0,
    ) -> pd.DataFrame:
        """
        Extract bus results for a scenario in gridfm-datakit `bus_data.parquet` layout.
        Delegates every column to a dedicated per-column function in
        `datakit_extractors` (see BUS_DATAKIT_COLUMNS). The three VLD status
        columns (BUS_BASE_STATUS / BUS_CONTINGENCY / BUS_STATUS) are appended
        as a superset so the topology-status supervision is preserved:
          - BUS_BASE_STATUS: healthy in the base network,
          - BUS_CONTINGENCY: directly-failed bus in this scenario,
          - BUS_STATUS: captures post-contingency energization, i.e.
          reachability from the REF bus through the healthy graph.
        """
        contingency_buses = {
            int(comp_idx) for comp_type, comp_idx in scenario if comp_type == "bus"
        }
        return dke.extract_bus_frame(
            net,
            scenario_id=scenario_id,
            load_scenario_idx=load_scenario_idx,
            base_mva=float(net.sn_mva),
            reachable_buses=reachable_buses,
            base_bus_status=base_bus_status,
            contingency_buses=contingency_buses,
            include_vld=True,
        )

    def _extract_gen_results(
        self,
        net,
        scenario_id: int,
        load_scenario_idx: int = 0,
    ) -> pd.DataFrame:
        """
        Extract generator results for a scenario in gridfm-datakit `gen_data.parquet` layout (one row per generator).
        This dataset is required by the foundation model's heterogeneous gen node type. Every column is
        produced by a dedicated `gen_col_*` function in `datakit_extractors`.
        """
        return dke.extract_gen_frame(
            net,
            scenario_id=scenario_id,
            load_scenario_idx=load_scenario_idx,
        )

    def _extract_branch_results(
        self,
        net,
        scenario_id: int,
        reachable_buses: Dict[int, bool],
        scenario: List[Tuple],
        base_line_status: Dict[int, int],
        base_trafo_status: Dict[int, int],
        load_scenario_idx: int = 0,
    ) -> pd.DataFrame:
        """
        Extract per-branch results for a scenario in gridfm-datakit
        `branch_data.parquet` layout (one row per physical line/trafo, with
        idx/from_bus/to_bus, flows, series params and full Yff/Yft/Ytf/Ytt).

        This replaces the previous bus-pair-template + aggregated G/B output.
        The three VLD status columns are appended as a superset:
          - BRANCH_BASE_STATUS: healthy in the base network,
          - BRANCH_CONTINGENCY: directly-failed branch in this scenario,
          - BRANCH_STATUS: energised (active and both endpoints reachable).
        """
        contingency_lines = {
            int(comp_idx) for comp_type, comp_idx in scenario if comp_type == "line"
        }
        contingency_trafos = {
            int(comp_idx) for comp_type, comp_idx in scenario if comp_type == "trafo"
        }
        return dke.extract_branch_frame(
            net,
            scenario_id=scenario_id,
            load_scenario_idx=load_scenario_idx,
            base_mva=float(net.sn_mva),
            reachable_buses=reachable_buses,
            base_line_status=base_line_status,
            base_trafo_status=base_trafo_status,
            contingency_lines=contingency_lines,
            contingency_trafos=contingency_trafos,
            line_active_fn=self._line_is_active,
            trafo_active_fn=self._trafo_is_active,
            include_vld=True,
        )

    def _compute_reachable_buses(self, net) -> Dict[int, bool]:
        """
        Build the healthy graph and mark buses reachable from the REF bus.

        Healthy graph rules:
        - buses must be in_service
        - line / trafo / bus-bus switch connections must be active
        """
        healthy_buses = {
            int(bus_idx): bool(net.bus.at[bus_idx, "in_service"])
            for bus_idx in net.bus.index
        }
        adjacency = {int(bus_idx): set() for bus_idx in net.bus.index}

        for line_idx, row in net.line.iterrows():
            if not self._line_is_active(net, line_idx):
                continue
            bus_i = int(row.from_bus)
            bus_j = int(row.to_bus)
            if healthy_buses.get(bus_i, False) and healthy_buses.get(bus_j, False):
                adjacency[bus_i].add(bus_j)
                adjacency[bus_j].add(bus_i)

        for trafo_idx, row in net.trafo.iterrows():
            if not self._trafo_is_active(net, trafo_idx):
                continue
            bus_i = int(row.hv_bus)
            bus_j = int(row.lv_bus)
            if healthy_buses.get(bus_i, False) and healthy_buses.get(bus_j, False):
                adjacency[bus_i].add(bus_j)
                adjacency[bus_j].add(bus_i)

        if len(net.switch) > 0:
            bus_bus_switches = net.switch[net.switch.et == "b"]
            for switch_idx, row in bus_bus_switches.iterrows():
                if not self._switch_is_active(net, switch_idx):
                    continue
                bus_i = int(row.bus)
                bus_j = int(row.element)
                if healthy_buses.get(bus_i, False) and healthy_buses.get(bus_j, False):
                    adjacency[bus_i].add(bus_j)
                    adjacency[bus_j].add(bus_i)

        reachable = {int(bus_idx): False for bus_idx in net.bus.index}
        ref_buses = [
            int(bus_idx)
            for bus_idx in net.ext_grid.bus.values
            if healthy_buses.get(int(bus_idx), False)
        ]
        stack = list(ref_buses)
        for bus_idx in ref_buses:
            reachable[bus_idx] = True

        while stack:
            bus_idx = stack.pop()
            for neighbor in adjacency.get(bus_idx, ()):
                if not reachable[neighbor]:
                    reachable[neighbor] = True
                    stack.append(neighbor)

        return reachable

    def _get_base_branch_status(self, net) -> Dict[Tuple[int, int], int]:
        return {
            (int(template["bus_i"]), int(template["bus_j"])): int(
                any(
                    self._component_is_active(net, comp_type, comp_idx)
                    for comp_type, comp_idx in template["components"]
                ),
            )
            for template in self.branch_templates
        }

    def _component_is_active(self, net, comp_type: str, comp_idx: int) -> bool:
        if comp_type == "line":
            return self._line_is_active(net, comp_idx)
        if comp_type == "trafo":
            return self._trafo_is_active(net, comp_idx)
        if comp_type == "switch_b":
            return self._switch_is_active(net, comp_idx)
        return False

    def _switch_is_active(self, net, switch_idx: int) -> bool:
        row = net.switch.loc[switch_idx]
        in_service = (
            bool(row["in_service"]) if "in_service" in net.switch.columns else True
        )
        return in_service and bool(row["closed"])

    def _line_is_active(self, net, line_idx: int) -> bool:
        if not bool(net.line.at[line_idx, "in_service"]):
            return False
        if len(net.switch) == 0:
            return True
        mask = (net.switch.et == "l") & (net.switch.element == line_idx)
        if not mask.any():
            return True
        switches = net.switch.loc[mask]
        in_service = (
            switches["in_service"] if "in_service" in switches.columns else True
        )
        return bool((switches["closed"] & in_service).all())

    def _trafo_is_active(self, net, trafo_idx: int) -> bool:
        if not bool(net.trafo.at[trafo_idx, "in_service"]):
            return False
        if len(net.switch) == 0:
            return True
        mask = (net.switch.et == "t") & (net.switch.element == trafo_idx)
        if not mask.any():
            return True
        switches = net.switch.loc[mask]
        in_service = (
            switches["in_service"] if "in_service" in switches.columns else True
        )
        return bool((switches["closed"] & in_service).all())

    def _determine_bus_type(self, net, bus_idx: int) -> Tuple[bool, bool, bool]:
        """Determine bus type (PQ, PV, REF)"""
        is_ref = bus_idx in net.ext_grid.bus.values

        is_pv = False
        if bus_idx in net.gen.bus.values:
            gen_mask = (net.gen.bus == bus_idx) & (net.gen.in_service)
            if gen_mask.any():
                is_pv = True

        is_pq = not (is_pv or is_ref)
        return is_pq, is_pv, is_ref

    def _get_bus_load(self, net, bus_idx: int) -> Tuple[float, float]:
        """Get total load at a bus"""
        load_mask = (net.load.bus == bus_idx) & (net.load.in_service)
        if load_mask.any():
            return net.load.loc[load_mask, "p_mw"].sum(), net.load.loc[
                load_mask,
                "q_mvar",
            ].sum()
        return 0.0, 0.0

    def _get_bus_generation(self, net, bus_idx: int) -> Tuple[float, float]:
        """Get total generation at a bus"""
        # From generators
        gen_mask = (net.gen.bus == bus_idx) & (net.gen.in_service)
        pg_mw = net.res_gen.loc[gen_mask, "p_mw"].sum() if gen_mask.any() else 0.0
        qg_mvar = net.res_gen.loc[gen_mask, "q_mvar"].sum() if gen_mask.any() else 0.0

        # From external grid
        ext_grid_mask = net.ext_grid.bus == bus_idx
        if ext_grid_mask.any():
            pg_mw += net.res_ext_grid.loc[ext_grid_mask, "p_mw"].sum()
            qg_mvar += net.res_ext_grid.loc[ext_grid_mask, "q_mvar"].sum()
        return pg_mw, qg_mvar

    def save_batch_results(
        self,
        bus_results: pd.DataFrame,
        gen_results: pd.DataFrame,
        branch_results: pd.DataFrame,
        batch_count,
    ):
        """
        Save batch results as partitioned parquet in gridfm-datakit layout.

        Writes three partitioned-parquet directories under raw_dir:
          - bus_data.parquet/    (scenario_partition=<k>/*.parquet)
          - gen_data.parquet/
          - branch_data.parquet/
        partitioned by `scenario_partition = scenario // N_SCENARIO_PER_PARTITION`.
        Each worker batch appends unique-named files, so parallel writes are safe
        and `pd.read_parquet(<dir>)` reconstructs the full dataset.
        """
        batch_num = batch_count
        dkw.save_batch_parquet(
            self.raw_dir,
            bus_results,
            gen_results,
            branch_results,
            batch_id=batch_num + 1,
        )
        n_scen = (
            len(bus_results) // len(self.net.bus)
            if (bus_results is not None and len(bus_results) > 0)
            else 0
        )
        print(
            f"Saved batch {batch_num + 1} ({n_scen} scenarios) as partitioned parquet",
        )

    def run_complete_contingency_analysis(self, max_n_level: int, max_size):
        """
        Run complete contingency analysis from N-1 to N-k with sequential batch numbering
        Parameters:
        max_n_level : int of the Maximum contingency level (1, 2, or 3)
        max_size : int, max memory per worker batch in MB; the number of
                   scenarios per batch is derived from it in
                   _calculate_optimal_batch_size (default 500 in the configs)
        """
        start_time = time.time()
        self.load_network_from_excel()
        if self.net is None:
            print("No network loaded")
            return
        self.global_batch_counter = 0  # Reset global batch counter
        total_batches_processed = 0  # Reset global processed batch counter
        global_scenario_counter = 0

        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_initializer,
            initargs=(
                self.network_file_path,
                self.flag,
                self.panda_net,
                self.output_dir,
                self.name,
            ),
        ) as executor:
            futures = []
            max_inflight = max_workers * 2  # bound number of in-flight batches

            for n_level in range(1, max_n_level + 1):
                print(f"Processing N-{n_level}")
                batch_generator = self.generate_contingency_scenarios(n_level, max_size)
                if batch_generator is None:
                    print(
                        "Failed to load scenario batches - check the size of the CSV file",
                    )
                    break
                for scenarios_batch in batch_generator:
                    while len(futures) >= max_inflight:
                        done = futures.pop(0)
                        done.result()  # ensure exceptions are raised and cleared
                    batch_index = self.global_batch_counter
                    futures.append(
                        executor.submit(
                            process_batch_worker,
                            n_level,
                            scenarios_batch,
                            batch_index,
                            global_scenario_counter,
                        ),
                    )
                    self.global_batch_counter += 1
                    global_scenario_counter += len(scenarios_batch)
                    total_batches_processed += 1
            for future in futures:
                future.result()  # drain remaining

        self.total_batches_processed = total_batches_processed
        self.time_taken = time.time() - start_time
        self._save_network_info()

    def _save_network_info(self):
        """Save network information file"""
        network_name = self.name
        info = {
            "time taken (s)": self.time_taken,
            "network_name": network_name,
            "contingency level": self.n_level,
            "number_scenarios": self.total_scenarios,
            "number_batches": self.total_batches_processed,
            "total_buses": len(self.net.bus),
            "total_lines": len(self.net.line),
            "total_generators": len(self.net.gen),
            "base_mva": self.net.sn_mva,
            "base_voltage_kv": self.net.bus.vn_kv.iloc[0]
            if len(self.net.bus) > 0
            else 0,
            "contingency_components": {
                "buses": len(self.component_indices["buses"]),
                "lines": len(self.component_indices["lines"]),
                "generators": len(self.component_indices["gens"]),
                "trafos": len(self.component_indices["trafos"]),
                "switches": len(self.component_indices["switches"]),
            },
        }

        info_file = os.path.join(self.output_dir, "network_info.txt")
        with open(info_file, "w") as f:
            for key, value in info.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")

    def get_results_summary(self) -> pd.DataFrame:
        """
        Summarise the generated partitioned-parquet datasets.

        Returns a DataFrame with one row per dataset (bus/gen/branch) giving
        row counts, scenario counts and on-disk size.
        """
        summary_data = []
        for name, subdir in (
            ("bus", dkw.BUS_DIR),
            ("gen", dkw.GEN_DIR),
            ("branch", dkw.BRANCH_DIR),
        ):
            path = os.path.join(self.raw_dir, subdir)
            if not os.path.isdir(path):
                continue
            try:
                df = dkw.read_dataset(self.raw_dir, name)
                n_rows = len(df)
                n_scen = df["scenario"].nunique() if n_rows else 0
            except Exception:
                n_rows, n_scen = 0, 0
            size_mb = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(path)
                for f in files
            ) / (1024 * 1024)
            summary_data.append(
                {
                    "dataset": name,
                    "path": subdir,
                    "rows": n_rows,
                    "scenarios": n_scen,
                    "size_mb": round(size_mb, 3),
                },
            )
        return pd.DataFrame(summary_data)

    def _assert_required_columns(
        self,
        df: pd.DataFrame,
        required_columns: set,
        name: str,
    ):
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    def _validate_binary_column(self, df: pd.DataFrame, column: str):
        values = set(
            pd.to_numeric(df[column], errors="coerce").dropna().unique().tolist(),
        )
        invalid = sorted(values - {0, 1})
        if invalid:
            raise ValueError(f"Column {column} contains non-binary values: {invalid}")

    def validate_generated_data(self, admittance_tol: float = 1e-10) -> Dict[str, int]:
        """
        Validate the generated gridfm-datakit partitioned-parquet datasets.

        Checks:
        - the base datakit + VLD columns exist for bus / gen / branch,
        - BUS_STATUS / BRANCH_STATUS are binary,
        - (scenario, bus) and (scenario, idx) keys are unique,
        - per-scenario row counts are constant,
        - scenario IDs align across bus / gen / branch,
        - obvious status / admittance / voltage inconsistencies are reported.
        """
        bus_required_columns = set(
            ["scenario"] + dke.BUS_DATAKIT_COLUMNS + dke.BUS_VLD_COLUMNS,
        )
        gen_required_columns = set(["scenario"] + dke.GEN_DATAKIT_COLUMNS)
        branch_required_columns = set(
            ["scenario"] + dke.BRANCH_DATAKIT_COLUMNS + dke.BRANCH_VLD_COLUMNS,
        )

        bus_df = dkw.read_dataset(self.raw_dir, "bus")
        gen_df = dkw.read_dataset(self.raw_dir, "gen")
        branch_df = dkw.read_dataset(self.raw_dir, "branch")

        self._assert_required_columns(bus_df, bus_required_columns, "bus_data.parquet")
        self._assert_required_columns(gen_df, gen_required_columns, "gen_data.parquet")
        self._assert_required_columns(
            branch_df,
            branch_required_columns,
            "branch_data.parquet",
        )
        self._validate_binary_column(bus_df, "BUS_STATUS")
        self._validate_binary_column(branch_df, "BRANCH_STATUS")

        duplicate_buses = int(bus_df.duplicated(subset=["scenario", "bus"]).sum())
        if duplicate_buses:
            raise ValueError(f"Found {duplicate_buses} duplicated (scenario, bus) rows")

        duplicate_branches = int(branch_df.duplicated(subset=["scenario", "idx"]).sum())
        if duplicate_branches:
            raise ValueError(
                f"Found {duplicate_branches} duplicated (scenario, idx) rows",
            )

        bus_counts = bus_df.groupby("scenario")["bus"].nunique()
        branch_counts = branch_df.groupby("scenario").size()
        if bus_counts.nunique() != 1:
            raise ValueError(
                "Bus count per scenario is not constant across the dataset",
            )
        if branch_counts.nunique() != 1:
            raise ValueError(
                "Branch count per scenario is not constant across the dataset",
            )

        bus_scenarios = set(bus_df["scenario"].unique().tolist())
        gen_scenarios = set(gen_df["scenario"].unique().tolist())
        branch_scenarios = set(branch_df["scenario"].unique().tolist())
        if bus_scenarios != branch_scenarios:
            raise ValueError("Scenario IDs differ between bus and branch datasets")
        if len(gen_df) and gen_scenarios != bus_scenarios:
            raise ValueError("Scenario IDs differ between bus and gen datasets")

        # Per-branch admittance magnitude (from the full Y-elements).
        y_mag = (
            branch_df[
                ["Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i"]
            ]
            .abs()
            .sum(axis=1)
        )
        off_branch_with_admittance = branch_df[
            (branch_df["BRANCH_STATUS"] == 0) & (y_mag > admittance_tol)
        ]
        off_bus_with_voltage = bus_df[
            (bus_df["BUS_STATUS"] == 0) & (bus_df["Vm"].abs() > admittance_tol)
        ]

        summary = {
            "scenarios": len(bus_scenarios),
            "buses_per_scenario": int(bus_counts.iloc[0]),
            "branches_per_scenario": int(branch_counts.iloc[0]),
            "gens_total_rows": int(len(gen_df)),
            "off_branches_with_nonzero_admittance": int(
                len(off_branch_with_admittance),
            ),
            "off_buses_with_nonzero_voltage": int(len(off_bus_with_voltage)),
        }
        # Write validation results to text file
        validation_file = os.path.join(self.output_dir, "validation.txt")
        with open(validation_file, "w") as f:
            for key, value in summary.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
        return summary
