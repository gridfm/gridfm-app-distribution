# YAML configuration reference

Every experiment is driven by one YAML file in `examples/config/`.


## Full example (current style)

```yaml
task:
  task_name: VoltageLossDetection
data:
  baseMVA: [1.0]          # one per network: base_mva in data/<network>/network_info.txt
  mask_value: 0.0
  normalization: HeteroDataMVANormalizer
  networks:
  - rural_MVLV_test
  scenarios:
  - 250000
  test_ratio: 0.1
  val_ratio: 0.1
  workers: 32
  split_by_load_scenario_idx: true
model:
  attention_head: 8
  edge_dim: 13
  hidden_size: 48
  input_bus_dim: 17
  input_gen_dim: 6
  output_bus_dim: 2
  output_gen_dim: 1
  num_layers: 12
  type: GNS_heterogeneous
optimizer:
  beta1: 0.9
  beta2: 0.999
  learning_rate: 0.0005
  lr_decay: 0.7
  lr_patience: 5
seed: 0
training:
  batch_size: 64
  epochs: 200
  loss_weights:
    - 0.1
    - 0.5
    - 0.4
  losses:
    - LayeredWeightedPhysics
    - MaskedBusMSE
    - VLDTopologyLoss
  loss_args:
    - { base_weight: 0.5 }
    - {}
    - {
        input_state_threshold: 0.5,
        prediction_threshold: 0.5,
        topology_weight: 1.0,
        target_anchor_weight: 0.25,
        off_vm_weight: 1.0,
        on_vm_weight: 0.5,
        unreachable_l1_weight: 1.0,
        topology_confidence_gamma: 10.0
      }
callbacks:
  patience: 100
  tol: 0
```

---

## Top-level keys

- `task`: task-specific settings (`OptimalPowerFlow` or `PowerFlow`).
- `data`: dataset selection, normalization, splits, and loading behavior.
- `model`: model architecture and dimensions.
- `optimizer`: optimizer and scheduler parameters.
- `training`: epochs, loss composition, and accelerator strategy.
- `callbacks`: early stopping behavior.
- `seed`: random seed used for reproducible shuffling/splits.
- `verbose`: enables extra outputs (for example additional test plots/log artifacts).

---

## `task` section

### `task.task_name`

Task name registered in the framework:

- `VoltageLossDetection`

### `task.ref_vm_input`, `task.mask_pv_vm` (VLD, optional)

These two flags control the VLD masking. Both default to `false`, which gives the original masking.

- `ref_vm_input: true` gives the slack bus Vm set-point as an input.
- `mask_pv_vm: true` predicts the PV bus Vm instead of copying its post-contingency input. That input is 0 at de-energised PV buses, so copying it leaks the answer.

Use the same values for training and evaluation. See [VLD Phase 1 fixes](../applications/vld_phase1_fixes.md).

---

## `data` section

### `data.networks`

List of dataset folders under your data root.
Examples: `rural_MVLV`, `urban_MVLV`, `suburban_MVLV`, `semi-urban_MVLV`.

### `data.scenarios`

List of scenario counts, one value per network in `data.networks`.
Example: with two networks, use two scenario entries in matching order.

### `data.normalization`

Normalizer class name:

- `HeteroDataMVANormalizer`: fit one normalization scale from training data.
- `HeteroDataPerSampleMVANormalizer`: fit per-scenario scales across the selected dataset.

### `data.baseMVA` (required, one value per network)

Each network's MVA base: the `sn_mva` of its pandapower net, which the data generator writes to `data/<network>/network_info.txt` as `base_mva`. Give a list in the order of `data.networks`, like `data.scenarios`:

```yaml
data:
  networks: [scenario_kerber_2, scenario_33radial, JRC_LV_urban]
  baseMVA:  [1.0,               100.0,             0.01]
```

A mapping (`baseMVA: {scenario_kerber_2: 1.0, ...}`) also works. A single number is accepted only when `data.networks` has one network; with several networks it is an error, because each network is normalised with its own base.

**How it is used.** The admittances (Yff/Yft) and shunts (GS/BS) in the data are in per unit on that base. Each network's normaliser converts them to its fitted power base with `base_mva / fitted_baseMVA`.

**The check.** When the datamodule is created, before any normaliser is built, fitted or restored, every value is compared with the network's `network_info.txt` (in the network folder or its `raw/` folder). One error lists every mismatch or missing file, and the run stops. Saved normaliser statistics (`--normalizer_stats`) must carry the same base, otherwise restoring them raises an error. Statistics from runs before this change carry the old common base 100.

**Pre-flight from a job script.** The same check is available as a command, so a Slurm job can stop before `srun`:

```bash
python -m gridfm_app_dist.datasets.network_info --config config/run_1.yaml --data_path data || exit 1
```

It prints the `baseMVA` list read from the `network_info.txt` files and a per-network table.

### `data.check_base_mva` (optional, default `true`)

`false` skips the comparison with `network_info.txt` and issues a warning. The format is still validated. Use it only to evaluate checkpoints trained with the legacy common base: set `baseMVA` to `100` for every network and use their original normaliser statistics.

### `data.mask_value`

Fill value used when masking unavailable measurements/features (examples use `0.0`).

### `data.test_ratio`, `data.val_ratio`

Fractions for validation and test splits when split files are not supplied.

### `data.workers`

Number of dataloader workers.

### `data.split_by_load_scenario_idx`

- `true`: split train/val/test by load scenario identifiers.
- `false`: perform standard random split.

### `data.split_from_existing_files`

Optional path to precomputed split files. When provided:

- split IDs are loaded from this folder,
- `data.scenarios` is ignored for split construction,
- do **not** combine with `split_by_load_scenario_idx: true`.

## `model` section

Current configs use the heterogeneous GNS model:

- `type`: model registry name (examples use `GNS_heterogeneous`).
- `input_bus_dim`: bus-node input feature dimension.
- `input_gen_dim`: generator-node input feature dimension.
- `output_bus_dim`: bus-node output dimension.
- `output_gen_dim`: generator-node output dimension.
- `edge_dim`: edge feature dimension.
- `hidden_size`: hidden feature width.
- `num_layers`: number of stacked message-passing layers.
- `attention_head`: attention head count per layer.
- `edge_input_transform` (optional, `none` | `asinh`, default `none`): `asinh(Y / edge_input_scale)` on the admittance columns that feed the edge MLP. The physics layers keep the raw values. A checkpoint must be evaluated with the setting it was trained with.
- `edge_input_scale` (optional, default `1.0`): $Y_0$ of the asinh transform.

---

## `training` section

### Core training controls

- `batch_size`: mini-batch size.
- `epochs`: number of epochs.
- `accelerator`: Lightning accelerator (`auto`, `mps`, `cpu`, `gpu`, etc.).
- `devices`: Lightning device selection (`auto`, integer, list).
- `strategy`: Lightning strategy (`auto`, `ddp`, etc.).
- `gradient_clip_val` (optional): clip the global gradient norm to this value. The default is `1.0` for `VoltageLossDetection` and off for the other tasks; `null` or `0` switches it off.
- `gradient_clip_algorithm` (optional, `norm` | `value`, default `norm`).
- `max_nonfinite_steps` (optional, default `20`): optimizer steps with a NaN/Inf loss or gradient are skipped. Training stops after this many in a row.

### Multi-loss configuration

- `losses`: list of registered loss names.
- `loss_weights`: scalar weight per loss.
- `loss_args`: list of argument objects matching `losses` by position.

All three lists must be aligned (same length and same order).

Registered loss names in current code:

- `LayeredWeightedPhysics`
- `MaskedGenMSE`
- `MaskedBusMSE`
- `QgViolationPenalty`
- `LossPerDim`
- `MaskedMSE`
- `MSE`
- `PBE`
- `VLDTopologyLoss`

Common `loss_args` patterns:

- `LayeredWeightedPhysics`: `{base_weight: <float>}`
- `LossPerDim`: `{dim: VM|VA|P_in|Q_in, loss_str: MAE|MSE}`
- `MaskedMSE`: `{channels: [Vm, Va, Qg, gen_Pg]}` (any subset; the default is `[Vm, Va]` for VLD and all four otherwise).
- `PBE`: `{slack_power: label|gen_rows, merge_switches: <bool>, bus_scope: all|energised, normalisation: none|load, switch_tol: <float>, include_shunts: <bool>}`. The VLD defaults are `label`, `true`, `energised`, `none`; for the other tasks they are `gen_rows`, `false`, `all`, `none` (the original loss). See [VLD Phase 1 fixes](../applications/vld_phase1_fixes.md).
- `VLDTopologyLoss`: see [VLD loss parameters](../applications/VLD_loss_parameters.md).
- `MaskedGenMSE`, `MaskedBusMSE`, `QgViolationPenalty`, `MSE`: `{}`

---

## `optimizer` section

- `learning_rate`: initial learning rate.
- `beta1`, `beta2`: Adam betas.
- `lr_decay`: scheduler decay factor (e.g., ReduceLROnPlateau factor).
- `lr_patience`: epochs to wait before applying LR decay.

---

## `callbacks` section

- `patience`: early stopping patience (epochs without sufficient improvement).
- `tol`: minimum required improvement threshold to reset patience.

---

## Practical validation checklist

Before launching a run, verify:

- `len(data.networks) == len(data.scenarios) == len(data.baseMVA)`, and each `baseMVA` equals the `base_mva` in that network's `network_info.txt` (`python -m gridfm_app_dist.datasets.network_info --config ... --data_path ...`).
- `len(training.losses) == len(training.loss_weights) == len(training.loss_args)`.
- `split_by_load_scenario_idx` and `split_from_existing_files` are not both active.
