# Scenario generator for gridfm_dist_app

Library to generate power-system contingency datasets in the parquet
format, so they can be consumed directly by the VLD task in `gridfm_dist_app`.

Given a network description (an Excel file at `network_file_path`, or an
in-memory pandapower network), the pipeline enumerates N-k contingencies, solves
a power flow for each, derives per-bus / per-generator / per-branch features and
topology-status labels, and writes three partitioned-parquet datasets:

```
<output_dir>/raw/
├── bus_data.parquet/      (scenario_partition=<k>/*.parquet)
├── gen_data.parquet/
└── branch_data.parquet/
```

---

## Quick start

```
# 1. Set tuning parameters in config.yaml, then run:
python run_scenarios.py --config config.yaml

# Variants
python run_scenarios.py --config config.yaml --dry-run     # validate config only
```

Each job in `config.yaml` maps onto the Scenario generator API:

```python
ScenarioGenerator(network_file_path, flag, net, output_dir) \
    .run_complete_contingency_analysis(max_n_level, max_size)
```

Dependencies: `pandapower`, `pandas`, `numpy`, `pyarrow`, `pyyaml`, `psutil`,
`tqdm` (and `numba` for faster power flow).

---

## The data-generation pipeline

The end-to-end flow lives in `ScenarioGenerator.run_complete_contingency_analysis`
(`scenario_data_gen.py`). The workflow in the pipeline follows the steps below.

### Step 0 — Inputs

The input network for which the data of contingency scenarios is being generated is loaded in the formats below:
1. **Excel network file** at `network_file_path`, loaded via one of three
  importers selected by `flag` (`1`, `2`, `3`). Each importer is customised to the structure of the network data in the excel file
2. **In-memory pandapower network** (`flag: null`), either passed in code or
  built from a `pandapower.networks` factory by name (`builtin_network:` in the
  config).

The number of contingencies and the parameters for the data generation are set:
```
max_n_level                 -> contingency depth
max_size                    -> batch sizing
n_scenario_per_partition    -> parquet partition size
output_dir                  -> output folder
```


### Step 1 — Load the network  (`load_network_from_excel`)

The chosen importer builds a pandapower `net` object with the standard element
tables: `bus`, `line`, `trafo`, `gen`, `ext_grid`, `load`, `switch`, `shunt`,
`poly_cost`. This is the single source of truth for all downstream features.
**Network loading and processing are intentionally kept inside
`ScenarioGenerator`** — only extraction/output was adapted to the datakit format.

### Step 2 — Index components  (`_index_components`)

Every contingency-eligible element is catalogued as a `(type, index)` tuple:
buses (excluding the reference/slack bus, which is never tripped), lines, gens,
trafos and switches. These form the universe from which contingency combinations
are drawn.

### Step 3 — Build branch templates  (`_build_branch_templates`)

A stable, bus-pair-keyed map of the physical elements (lines / trafos / bus-bus
switches) connecting each pair of buses. This keeps branch bookkeeping stable
across scenarios and underpins the reachability graph.

### Step 4 — Enumerate contingency scenarios  (`generate_contingency_scenarios`)

For each level `n` from `1` to `max_n_level`, all N-k combinations of components
are generated with `itertools.combinations` (there are `C(n_components, n)` of
them). Scenarios are **yielded in batches**; `_calculate_optimal_batch_size`
picks a batch size from `max_size` and the per-scenario memory footprint so the
run stays memory-bounded.

### Step 5 — Solve each scenario  (`run_batch_contingency_analysis`)

For every scenario in a batch:

1. **Reset** the network to its base in-service state (cached once per batch).
2. **Apply the contingency** (`apply_contingency`): set the tripped elements out
   of service. Tripping a **bus** cascades — its loads, gens, incident lines and
   trafos go out of service and its switches open (`_disable_bus_components`).
3. **Protect the slack**: any ext_grid bus that got disabled is forced back in
   service (the reference must exist for a power flow to run).
4. **Run the power flow** with `pp.runpp`. If it errors or **does not converge**,
   the scenario is **skipped** (only physically solvable states are kept).
5. **Compute energisation** (`_compute_reachable_buses`): build the *healthy
   graph* (in-service buses connected by active lines/trafos/closed switches)
   and BFS outward from the reference bus. A bus is "energised" only if it is
   reachable from the slack through healthy equipment — this is the ground-truth
   topology label the VLD task learns.

### Step 6 — Extract features per column  (`datakit_extractors.py`)

From the solved network, three frames are built, **one dedicated function per
column**, in the exact datakit schema order:

- **`bus`** — loads (`Pd,Qd`), generation (`Pg,Qg`), voltages (`Vm,Va`),
  bus-type one-hots (`PQ,PV,REF`), base voltage & limits (`vn_kv,
  min/max_vm_pu`), shunt (`GS,BS`).
- **`gen`** — dispatch (`p_mw,q_mvar`), limits, cost coefficients
  (`cp0/cp1/cp2`), `in_service`, `is_slack_gen`.
- **`branch`** — per line/trafo: endpoints (`from_bus,to_bus`), flows
  (`pf,qf,pt,qt`), series/shunt params (`r,x,b`), full complex admittance
  (`Yff/Yft/Ytf/Ytt`), `tap,shift,ang_min/max,rate_a,br_status`.

Each frame also carries `scenario`, `load_scenario_idx`, plus **VLD status
columns** appended as a superset: `BUS_BASE_STATUS / BUS_CONTINGENCY /
BUS_STATUS` and the `BRANCH_*` equivalents. (`BUS_STATUS` = energised;
`*_CONTINGENCY` = directly tripped; `*_BASE_STATUS` = healthy in the base case.)

See `DATAKIT_ADAPTATION.md` for the full column list, provenance, and admittance
formulas.

### Step 7 — Write partitioned parquet  (`datakit_writer.py`)

Each batch's bus/gen/branch frames are appended to their dataset directory,
Hive-partitioned by `scenario_partition = scenario // n_scenario_per_partition`
(datakit default 200). Parallel workers write uniquely-named files, so
`pd.read_parquet("<raw_dir>/bus_data.parquet")` transparently reconstructs the
full dataset — exactly what the model's dataloader does.

### Parallelism & orchestration

`run_complete_contingency_analysis` drives all of the above across a
`ProcessPoolExecutor` (workers = `SLURM_CPUS_PER_TASK`, default 8). Each worker
loads its own copy of the network once (`worker_initializer`) and processes whole
batches, bounding the number of in-flight batches to control memory. When it
finishes it writes `network_info.txt` with run metadata.

```
network_file_path / builtin net
        │  load (flag 1/2/3 or in-memory)
        ▼
   pandapower net ──► index components ──► branch templates
        │
        ▼  for n = 1..max_n_level
   enumerate N-k combinations ──► batch
        │  (parallel workers)
        ▼  per scenario
   reset ─► apply contingency ─► runpp ─► (skip if not converged)
        │
        ▼
   reachability BFS  +  per-column extraction (bus / gen / branch)
        │
        ▼
   partitioned parquet  ─►  <output_dir>/raw/{bus,gen,branch}_data.parquet/
```

---

## Configuration (`config.yaml`)

`defaults:` are applied to every job; each `scenarios:` entry can override them.

| Key | Meaning |
|---|---|
| `name` | Unique job name (required) |
| `network_file_path` | Excel network file (required when `flag` is 1/2/3) |
| `flag` | Excel importer method `1`/`2`/`3`, or `null` for an in-memory net |
| `builtin_network` | For `flag: null`, a `pandapower.networks` factory name (e.g. `case33bw`) |
| `output_dir` | Output root (required); parquet goes to `<output_dir>/raw/` |
| `max_n_level` | Contingency depth: `1` = N-1, up to `3` |
| `max_size` | Target scenarios per worker batch |
| `n_scenario_per_partition` | Parquet partition size (datakit default 200) |
| `run_summary` / `run_validation` | Print dataset summary / validation after the job |
| `continue_on_error` | Keep running remaining jobs if one fails |

The runner validates the whole config up front (required keys, duplicate names,
overlapping `output_dir`s, Excel-vs-builtin consistency) and prints a final
succeeded/failed roll-up. See `config.yaml` for worked examples.

---

## Outputs

```
<output_dir>/
├── raw/
│   ├── bus_data.parquet/scenario_partition=<k>/*.parquet
│   ├── gen_data.parquet/scenario_partition=<k>/*.parquet
│   └── branch_data.parquet/scenario_partition=<k>/*.parquet
├── processed/            # reserved for the model's cached .pt tensors
└── network_info.txt      # run metadata (timing, counts, contingency level)
```

Read a dataset back with either pandas or the helper:

```python
import pandas as pd, os
bus = pd.read_parquet(os.path.join(raw_dir, "bus_data.parquet"))

# or
import datakit_writer as dkw
bus = dkw.read_dataset(raw_dir, "bus")   # "bus" | "gen" | "branch"
```

`scenario` identifies the contingency case; `load_scenario_idx` indexes the load
sample within a case (currently `0`, since one load snapshot is produced per
topology).

---

## Validation

After each job the runner calls `validate_generated_data`, which checks that all
datakit + VLD columns exist, status columns are binary, `(scenario, bus)` and
`(scenario, idx)` keys are unique, per-scenario counts are constant, and scenario
IDs align across bus/gen/branch. It also reports diagnostics such as
`off_branches_with_nonzero_admittance` — **expected**, because a de-energised
branch still has its static physical admittance even though its flows are zero.

---
