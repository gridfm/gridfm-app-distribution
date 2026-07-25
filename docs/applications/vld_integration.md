# Integrating the Voltage-Loss-Detection (VLD) Task into gridfm-graphkit

This document describes the changes made to [`gridfm-graphkit`](https://github.com/gridfm/gridfm-graphkit) to
integrate the new **Voltage-Loss-Detection (VLD)** task in the new
(`gridfm_dist_app`) in the `gridfm-app-distribution` package is laid out.

## Task summary 
The VLD task teaches the heterogeneous GNS to predict, per bus, whether that bus
remains energized after a set of component failures (contingencies), and to
predict a consistent voltage magnitude for the energized buses. This turns the
model into a topology-aware "will this bus stay live?" predictor on top of the
existing power-flow reconstruction backbone.



---

## 1. New feature columns (`datasets/globals.py`)

The heterogeneous feature layout was extended. Base graphkit indices are kept;
VLD adds four new indicators and repurposes the bus-output tail.

**Bus input features (added):**

| Constant | Index | Meaning |
|---|---|---|
| `BUS_BASE_STATUS_H` | 15 | Bus on/off status in the pre-contingency (base) topology |
| `BUS_CONT_H` | 16 | Bus directly hit by a contingency (failure) |

**Branch / edge features (added):**

| Constant | Index | Meaning |
|---|---|---|
| `BRANCH_BASE_STATUS_E` | 11 | Branch on/off status in the base topology |
| `BRANCH_CONT_E` | 12 | Branch directly hit by a contingency |

**Bus output layout (changed for VLD).** Base tasks emit
`[Vm, Va, Pg, Qg, Pd, Qd]` (6 cols, `PD_OUT=4`, `QD_OUT=5`). The VLD model emits
`[Vm, Va, Pg, Qg, status_logit]` (5 cols):

| Constant | Index | Meaning |
|---|---|---|
| `PHYSICAL_BUS_DIM` | 4 | Number of physical bus outputs `[Vm, Va, Pg, Qg]` |
| `BUS_STATUS_LOGIT_OUT` | 4 | Column holding the predicted status logit |
| `BUS_STATUS_TARGET` | 5 | Target column (energized/de-energized) in the target tensor |

> **Important consequence.** Because column 4 now means *status logit* (not
> `Pd`), any code that width-guards on `size(1) > PD_OUT (4)` would wrongly treat
> the status logit as a load-power column. This is why the normalizers and
> `PBELoss` were changed to guard on `BUS_STATUS_TARGET (5)` — see the code
> review, item 4.

## 2. New transforms (`datasets/transforms.py`, `datasets/task_transforms.py`)

- `LoadGridParamsFromPathVLD` — loads grid parameters plus base status and
  contingency indicators for buses and branches.
- `RemoveInactiveBranchesKeepTopology` — drops electrically inactive branches for
  the physics computation while preserving the connectivity graph used for the
  reachability labels.
- `VoltageLossDetectionTransforms` (registered under `TRANSFORM_REGISTRY` as
  `"VoltageLossDetection"`) — the task `Compose` pipeline:
  `LoadGridParamsFromPathVLD → RemoveInactiveBranchesKeepTopology → AddVLDHeteroMask`.

## 3. New masking (`datasets/masking.py`)

- `AddVLDHeteroMask` — produces the hetero mask dictionary for the VLD task,
  including the bus-status supervision mask.

## 4. New loss (`training/loss.py`)

- `VLDTopologyLoss` (registered `"VLDTopologyLoss"`). Objective =
  topology BCE + target-anchor BCE + off-bus Vm penalty + on-bus Vm penalty:
  - **Reachability labels** via `_build_graph_reachability`: iterative BFS from
    buses that are base-on and not contingency-hit, propagating over branches
    that are base-on and not contingency-hit — a direct encoding of the physical
    rule above.
  - **Topology BCE** trains the status logit against the hard reachability
    label, weighted by a confidence term
    `exp(-γ·|bus_available − reachable|)` that down-weights ambiguous buses.
  - **Target-anchor BCE** additionally anchors the logit to the supplied
    post-contingency status target.
  - **Off-Vm loss** drives predicted `Vm → 0` on unreachable (de-energized)
    buses (L2 + weighted L1).
  - **On-Vm loss** matches predicted `Vm` to the target on reachable buses,
    scaled by the (detached) predicted status probability.
  - Configurable via `loss_args`: `topology_weight`, `target_anchor_weight`,
    `off_vm_weight`, `on_vm_weight`, `unreachable_l1_weight`,
    `topology_confidence_gamma`, `input_state_threshold`, `prediction_threshold`.
- `PBELoss` (`"PBE"`) and `QgViolationPenaltyLoss` (`"QgViolationPenalty"`)
  carried from the reference and adapted to the VLD 6-argument loss call
  (they take `x_dict=None` and skip the shunt term when it is absent).

The loss reads topology from `model.latest_x_dict["bus"]` (stored by the GNS in
its forward pass) rather than receiving `x_dict` as a call argument — see the
interface note below.

## 5. New physics decoder (`tasks/utils.py`)

- `PhysicsDecoderVLD` (registered `"VoltageLossDetection"` in
  `PHYSICS_DECODER_REGISTRY`) joins the existing `PhysicsDecoderOPF/PF/SE`.
  **All four physics decoders are now registered in `tasks/utils.py`** (moved out
  of `models/utils.py` to avoid a duplicate-key crash).

## 6. New task module (`tasks/vld_task.py`)

- `VoltageLossDetectionTask(ReconstructionTask)` (registered
  `"VoltageLossDetection"` in `TASK_REGISTRY`). Adds VLD-specific test/predict
  metrics: status accuracy, off/on-bus Vm behaviour, and per-bus status
  probability outputs for downstream evaluation.

## 7. Model head (`models/gnn_heterogeneous_gns.py`)

- The heterogeneous GNS gains a dedicated `bus_status_head`. When
  `task == "VoltageLossDetection"` the head's `status_logit` is concatenated to
  the physical bus outputs, yielding the 5-column VLD bus output. The model also
  stores `latest_x_dict` in its forward pass so the loss can recover the input
  topology.

---

### Loss ↔ task interface

The VLD `shared_step` calls the loss **without** `x_dict`
(`loss_fn(pred, target, edge_index, edge_attr, mask, model=self.model)`), unlike
the base graphkit 7-argument interface. `MixedLoss.forward` was updated to match,
and the carried-over base losses accept `x_dict=None` so they remain callable in
this 6-argument path.

---

## Package layout (`gridfm-app-distribution`)

```
gridfm-app-distribution/
├── gridfm_dist_app/
│   ├── __init__.py            # imports datasets, tasks, models to populate registries
│   ├── __main__.py
│   ├── cli.py
│   ├── datasets/
│   │   ├── __init__.py        # registers normalizers + task transforms (incl. VLD)
│   │   ├── globals.py         # feature indices (+ VLD additions)
│   │   ├── powergrid_hetero_dataset.py
│   │   ├── hetero_powergrid_datamodule.py
│   │   ├── transforms.py      # incl. LoadGridParamsFromPathVLD, RemoveInactiveBranchesKeepTopology
│   │   ├── task_transforms.py # incl. VoltageLossDetectionTransforms
│   │   ├── masking.py         # incl. AddVLDHeteroMask
│   │   ├── normalizers.py     # VLD-safe inverse_output guards
│   │   ├── utils.py
│   │   └── postprocessing.py
│   ├── tasks/
│   │   ├── __init__.py        # registers PF/OPF/SE/VLD tasks
│   │   ├── base_task.py
│   │   ├── reconstruction_tasks.py
│   │   ├── pf_task.py · opf_task.py · se_task.py
│   │   ├── vld_task.py        # VoltageLossDetectionTask
│   │   ├── utils.py           # all 4 physics decoders + plotting helpers
│   │   └── compute_ac_dc_metrics.py  # compatibility stub (non-VLD)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_heterogeneous_gns.py  # + bus_status_head, latest_x_dict
│   │   └── utils.py           # shared physics blocks (decoders moved to tasks/utils.py)
│   ├── io/
│   │   ├── __init__.py
│   │   ├── registries.py
│   │   ├── param_handler.py
│   │   └── visualization.py
│   └── training/
│       ├── __init__.py
│       ├── loss.py            # + VLDTopologyLoss, PBE, QgViolationPenalty
│       └── callbacks.py
├── docs/
│   ├── CODE_REVIEW.md
│   └── INTEGRATION_CHANGES.md
└── examples/config/vld_config.yaml
```

Every `gridfm_graphkit.*` import in the ported files was rewritten to
`gridfm_dist_app.*`.

## Example configuration

See `examples/config/vld_config.yaml`. The key VLD settings are:

```yaml
task:
  task_name: VoltageLossDetection
model:
  type: GNS_heterogeneous
  output_bus_dim: 5          # [Vm, Va, Pg, Qg, status_logit]
training:
  losses: ["VLDTopologyLoss"]
  loss_weights: [1.0]
  loss_args:
    - topology_weight: 1.0
      target_anchor_weight: 0.25
      off_vm_weight: 1.0
      on_vm_weight: 0.5
      unreachable_l1_weight: 1.0
      topology_confidence_gamma: 10.0
      input_state_threshold: 0.5
      prediction_threshold: 0.5
```
