# Contingency data generation

`contingency_data_gen` is the data-generation application of this repository. It
builds a pandapower network, enumerates N-1 … N-k contingencies (bus and branch
outages), solves each post-contingency power flow and writes the results in the
gridfm-datakit partitioned-parquet layout that `gridfm_app_dist` trains on.

## Workflow

```
contingency_data_gen --config data_gen_1.yaml
gridfm_app_dist train --config configs/scenario_33radial.yaml --data_path data
```

Each scenario in the config produces `<output_dir>/raw/{bus,gen,branch}_data.parquet`,
partitioned by `scenario_partition = scenario // n_scenario_per_partition`. With
the shipped configs `output_dir` is `data/<name>`, so run from the repository
root and point training at `--data_path data`.

## Config format

```yaml
defaults:
  flag: 1                   # 1/2/3 Excel import method, 9 netlib YAML, 10 MATPOWER, null pandapower factory
  max_n_level: 3            # contingency depth: N-1 … N-max_n_level
  max_size: 500             # max memory per worker batch (MB); the scenario count per batch is derived from it
  n_scenario_per_partition: 200
  run_summary: true
  run_validation: true
  continue_on_error: true

scenarios:
  - name: case33radial
    network_file_path: networks/case33radial.xlsx
    flag: 1
    output_dir: data/scenario_33radial
    max_n_level: 3
  - name: urban_LV
    flag: 9
    builtin_network: netbuilder/urban_LV.yaml
    output_dir: data/scenario_urban_LV
  - name: cigre_lv
    flag: null
    builtin_network: create_cigre_network_lv
    output_dir: data/scenario_CIGRE_LV
```

Relative `network_file_path` / `builtin_network` values are resolved against the
config file's directory first, then against the package `configs/` folder and
its `networks/` and `netbuilder/` sub-folders. A legacy `./config/` prefix is
accepted.

## Command line

| option | meaning |
|---|---|
| `--config PATH\|NAME` | a YAML file, or the name of a shipped config (`--list-configs`) |
| `--only NAME …` | run a subset of the scenarios in the config |
| `--dry-run` | validate the config and report missing input files without running |
| `--list-configs` | list the configs shipped in `contingency_data_gen/configs/` |

Worker count follows `SLURM_CPUS_PER_TASK` (default 8).

## Package layout

| module | role |
|---|---|
| `contingency_data_gen.generation.scenario_data_gen` | `ScenarioGenerator`: contingency enumeration, parallel power flows, parquet output, validation |
| `contingency_data_gen.generation.datakit_extractors` / `datakit_writer` | gridfm-datakit column extraction and partitioned writing |
| `contingency_data_gen.generation.excel_pandapower*` | three Excel-to-pandapower import methods (`flag` 1/2/3) |
| `contingency_data_gen.netlib` | `NetworkBuilder` and the standard line/transformer type catalogue for YAML-defined MV/LV networks |
| `contingency_data_gen.configs` | shipped job configs, Excel networks (`networks/`), netlib network definitions (`netbuilder/`) |

Network-builder examples and legacy comparison tools live in `examples/datakit/`.
