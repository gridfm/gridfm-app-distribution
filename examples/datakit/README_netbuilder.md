# Pandapower network builder

Builds and validates a pandapower network using network parameters in a yaml file used in
M. Kisuule, M. B. Ndawula and I. Hernando-Gil, "Distribution Network Reliability Equivalent Method using Impact Aggregation for Risk and Reliability Analysis," in Journal of Modern Power Systems and Clean Energy, doi: 10.35833/MPCE.2025.000893.


The network builder is a self-contained class `NetworkBuilder` housed in `netlib` library.
Running `NetworkBuilder.build()` returns the pandapower `net`.

Data on the network parameters is fed to a yaml file `configs/*.yaml`

## Structure of the library

```
pandapower_netbuilder/
├── netlib/
│   ├── __init__.py            # public API
│   ├── network_builder.py     # NetworkBuilder class (self-contained)
│   └── std_libraries.py       # line / transformer std type catalogue
├── build_network.py           # CLI
├── example_usage.py           # usage snippets
└── tools/
    ├── export_s1_to_yaml.py   # migration helper (legacy S1 -> YAML)
    └── test_equivalence.py    # regression test vs. the legacy scripts
```

## Usage

```python
from netlib import NetworkBuilder

builder = NetworkBuilder.from_yaml("configs/urban_MVLV.yaml")
net = builder.build()                 # -> pandapower net

builder.summary()                     # element counts
builder.report.as_dict()              # build time, diagnostic, convergence
builder.run_powerflow()
builder.export("Results")             # network JSON + report + results workbook
```


CLI:

```bash
python build_network.py configs/*.yaml --powerflow --export Results
python build_network.py configs/urban_MVLV.yaml --strict     # fail on diagnostic issues
```

Requirements: `pandapower`, `pyyaml` (+ `pandas`/`openpyxl` for the Excel
export, `numba` for faster power flows).

## YAML schema

```yaml
scenario: Full MV Network - Generic Urban Network

std_types:                 # names resolved from netlib.std_libraries.STD_TYPES
  line:  [P_185, Q_95, A_300, ...]
  trafo: [trafo_33_11_15, trafo_11_400_500]

std_type_definitions:      # optional: inline parameters, override the catalogue
  line:
    X_240: {r_ohm_per_km: 0.125, x_ohm_per_km: 0.08, c_nf_per_km: 60.0,
            max_i_ka: 0.45, type: cs}

mv:
  v_ref: 1.0               # slack voltage in pu
  slack_bus: 1             # bus index of the external grid
  sections:                # number of subsections per section type
    main_substation: 1
    main_trunk_feeder: 0
    feeder: 6
  voltage_levels: {MV_33kV: 33.0, MV_11kV: 11.0}
  bus_counts:              # one list entry per subsection, per voltage level
    MV_33kV_main_substation: [3]
    MV_33kV_main_trunk_feeder: []
    MV_33kV_feeder: []
    MV_11kV_main_substation: [9]
    MV_11kV_main_trunk_feeder: []
    MV_11kV_feeder: [17, 17, 17, 17, 17, 17]
  feeder_buses: [7, 8, 9, 10, 11, 12]      # MV bus supplying each feeder
  circuit_breakers:                         # section -> [[from_bus, to_bus], ...]
    Main_substation: [[1, 2], [1, 3]]
    feeder1: [[71, 72], [73, 74]]
    LV: [[null, 0]]        # reserved section: replicated in every LV network,
                           # null = the secondary substation MV bus itself
  fuses:                   # same structure, created as type "LS"
    feeder5: [[151, 152]]
  lines:                   # std_type -> length_km -> section -> [[from, to], ...]
    P_185:
      0.375:
        feeder1: [[7, 71], [71, 73]]
  trafos:                  # std_type -> [[hv_bus, lv_bus], ...]
    trafo_33_11_15: [[2, 4], [3, 5]]

lv:                        # template of ONE LV network, local bus numbers
  buses_mv_per_network: 1
  buses_lv_per_network: 52
  voltage_levels: {MV_11kV: 11.0, LV: 0.4}
  sec_substations: [72, 74, 76, ...]        # MV buses carrying an LV network
  sec_substations_reduced: []               # MV buses with a lumped equivalent
  sec_substations_reduced_main_trunk_feeder: []
  feeder_buses_lvnetworks: []               # LV networks per reduced feeder bus
  circuit_breakers: [[2, 3], [2, 8]]        # local bus pairs
  fuses: [[1, 2]]
  lines:                   # std_type -> length_km -> [[from, to], ...]
    A_300:
      0.032: [[2, 19]]
  trafos:
    trafo_11_400_500: [[0, 1]]

load_points:
  lp_per_customer: 0.00227 # ADMD in MW
  power_factor: 0.98
  loads:
    - {bus: 6, customers: 8}

options:
  lp_placement: [lv]       # lv | mv_reduced | feeder_eq | trunk_feeder (list ok)
  run_diagnostic: true
  strict_diagnostic: false # true -> raise NetworkDiagnosticError on any issue
  quiet_diagnostic: true
  run_powerflow: false
  mv_bus_index_start: 1    # or "first_feeder_bus" (legacy feeder-equivalent mode)
```

### Load point placement

| `lp_placement` | loads created at |
|---|---|
| `lv` | every bus of every replicated LV network (`lv.sec_substations`) |
| `mv_reduced` | `lv.sec_substations_reduced` (lumped equivalents) |
| `feeder_eq` | `lv.sec_substations_reduced`, one load set per `feeder_buses_lvnetworks` entry |
| `trunk_feeder` | `lv.sec_substations_reduced_main_trunk_feeder` |

## Adding a new network

1. Copy the closest YAML file in `configs/` and edit the data.
2. Add any new conductor / transformer to `netlib/std_libraries.py` (or define
   it inline under `std_type_definitions`).
3. `python build_network.py configs/my_network.yaml --powerflow`.


## Validation

`NetworkBuilder` validates the input before building and raises
`NetworkConfigError` for: missing `mv` block, missing `bus_counts` entry for a
voltage level / section combination, more feeder subsections than
`feeder_buses`, unknown std type names, unknown `lp_placement`, and LV networks
without `buses_lv_per_network`.

`tools/test_equivalence.py` rebuilds each scenario with the original S1/S2 code
and compares buses, lines, transformers, switches, loads and the external grid
element by element, plus the std type parameters against `Std_libraries.py`:

```bash
python tools/test_equivalence.py /path/to/legacy_scripts
# 7/7 checks passed
```
