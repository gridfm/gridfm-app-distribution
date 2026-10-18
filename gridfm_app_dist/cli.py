from gridfm_app_dist.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_app_dist.io.param_handler import NestedNamespace
from gridfm_app_dist.io.registries import DATASET_WRAPPER_REGISTRY
from gridfm_app_dist.training.callbacks import SaveBestModelStateDict
import importlib
import numpy as np
import os
import time
import yaml
import torch
import pandas as pd

from gridfm_app_dist.io.param_handler import get_task
from gridfm_app_dist.tasks.compute_ac_dc_metrics import compute_ac_dc_metrics
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import lightning as L


def _load_plugins(plugins: list[str]) -> None:
    """Import plugin packages so their registry decorators fire."""
    for plugin_pkg in plugins:
        try:
            importlib.import_module(plugin_pkg)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Plugin package '{plugin_pkg}' could not be imported: {e}. "
                "Make sure it is installed in the current environment.",
            ) from e


def _validate_dataset_wrapper(name: str | None) -> None:
    """Raise a helpful error if *name* is not registered in DATASET_WRAPPER_REGISTRY."""
    if name is None:
        return
    if name not in DATASET_WRAPPER_REGISTRY:
        available = list(DATASET_WRAPPER_REGISTRY)
        raise KeyError(
            f"Dataset wrapper '{name}' is not registered. "
            f"Available wrappers: {available}. "
            "If it lives in a plugin package, pass it via --plugins.",
        )


def benchmark_cli(args):
    """Benchmark train-dataloader iteration speed over one or more epochs."""
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    config_args = NestedNamespace(**base_config)

    num_workers_override = getattr(args, "num_workers", None)
    if num_workers_override is not None:
        config_args.data.workers = num_workers_override

    _load_plugins(getattr(args, "plugins", []))

    dataset_wrapper = getattr(args, "dataset_wrapper", None)
    dataset_wrapper_cache_dir = getattr(args, "dataset_wrapper_cache_dir", None)
    _validate_dataset_wrapper(dataset_wrapper)

    print("Setting up datamodule...")
    t0 = time.perf_counter()
    dm = LitGridHeteroDataModule(
        config_args,
        args.data_path,
        dataset_wrapper=dataset_wrapper,
        dataset_wrapper_cache_dir=dataset_wrapper_cache_dir,
    )
    dm.setup(stage="fit")
    setup_time = time.perf_counter() - t0
    print(f"  Setup time        : {setup_time:.2f}s")

    loader = dm.train_dataloader()
    num_batches = len(loader)
    print(f"  Train batches     : {num_batches}")
    print(f"  Batch size        : {config_args.training.batch_size}")
    print(f"  Workers           : {config_args.data.workers}")
    print(f"  Dataset wrapper   : {dataset_wrapper or 'none'}")
    print()

    epoch_times = []
    for epoch in range(args.epochs):
        t_start = time.perf_counter()
        for _batch in loader:
            pass
        elapsed = time.perf_counter() - t_start
        per_batch = elapsed / num_batches if num_batches > 0 else 0.0
        epoch_times.append(elapsed)
        print(
            f"Epoch {epoch:>3}: {elapsed:7.3f}s total  "
            f"{per_batch:.4f}s/batch  ({num_batches} batches)",
        )

    if args.epochs > 1:
        avg = sum(epoch_times) / len(epoch_times)
        print(f"\nAverage over {args.epochs} epochs: {avg:.3f}s")


class PhysicsGateMonitor(L.Callback):
    """Track physics_scale so a receptive-field collapse cannot pass unnoticed.

    Parameters
    ----------
    floor : float
        Value below which a layer's gate is considered closed. 0.10 is the point
        at which measured reach at num_layers=24 fell below the 34 hops the
        deepest training network needs.
    per_layer : bool
        Log every layer individually as well as the summary statistics.
    """

    def __init__(self, floor: float = 0.10, per_layer: bool = True):
        self.floor = float(floor)
        self.per_layer = bool(per_layer)
        self._warned = False
        self._missing_logged = False

    @staticmethod
    def _gate(pl_module):
        model = getattr(pl_module, "model", None)
        scale = getattr(model, "physics_scale", None)
        # None when physics_gate: false -- the ungated branch has no LayerScale.
        return scale if isinstance(scale, torch.Tensor) else None

    def on_train_start(self, trainer, pl_module):
        gate = self._gate(pl_module)
        if not trainer.is_global_zero:
            return
        if gate is None:
            print(
                "[gate] physics_scale absent (physics_gate: false); "
                "receptive field is 1 hop/layer",
                flush=True,
            )
            return
        vals = gate.detach().float().cpu()
        print(
            f"[gate] physics_scale: {gate.numel()} layers, "
            f"init min={vals.min():.4f} mean={vals.mean():.4f} max={vals.max():.4f}, "
            f"trainable={gate.requires_grad}",
            flush=True,
        )
        if vals.max() <= 0.0:
            print(
                "[gate] WARNING: gate starts fully closed. The physics hop is "
                "inactive, so reach is num_layers hops, not ~1.75x that.",
                flush=True,
            )

    def on_train_epoch_end(self, trainer, pl_module):
        gate = self._gate(pl_module)
        if gate is None:
            if trainer.is_global_zero and not self._missing_logged:
                self._missing_logged = True
            return

        vals = gate.detach().float().cpu()
        below = float((vals.abs() < self.floor).float().mean())
        metrics = {
            "gate/mean": float(vals.mean()),
            "gate/min": float(vals.min()),
            "gate/max": float(vals.max()),
            "gate/abs_min": float(vals.abs().min()),
            "gate/frac_below_floor": below,
        }
        if self.per_layer:
            for i, v in enumerate(vals.tolist()):
                metrics[f"gate/layer_{i:02d}"] = float(v)

        # Use log_metrics rather than pl_module.log
        # to keep this out of the epoch-end reduction / sync_dist path.
        if trainer.is_global_zero and trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        if trainer.is_global_zero and below > 0.5 and not self._warned:
            self._warned = True
            print(
                f"[gate] WARNING: {below:.0%} of layers have |physics_scale| < "
                f"{self.floor}. The physics hop is collapsing and the effective "
                f"receptive field is approaching num_layers hops. If deep-bus "
                f"status accuracy stalls, clamp the gate or freeze it.",
                flush=True,
            )

    def on_train_end(self, trainer, pl_module):
        gate = self._gate(pl_module)
        if gate is None or not trainer.is_global_zero:
            return
        vals = gate.detach().float().cpu()
        print(
            f"[gate] final physics_scale: min={vals.min():.4f} "
            f"mean={vals.mean():.4f} max={vals.max():.4f} "
            f"({float((vals.abs() < self.floor).float().mean()):.0%} below "
            f"{self.floor})",
            flush=True,
        )


def get_training_callbacks(args):
    early_stop_callback = EarlyStopping(
        monitor="Validation loss",
        min_delta=args.callbacks.tol,
        patience=args.callbacks.patience,
        verbose=False,
        mode="min",
    )

    # For VLD the total loss is dominated by the voltage / PBE terms, so the
    # lowest-loss epoch is not necessarily the best classifier. Select the
    # exported state dict on balanced accuracy (mean of off- and on-recall,
    # computed at the loss's own 1/(1+w_off) threshold) instead. Early stopping
    # stays on the loss.
    if getattr(args.task, "task_name", None) == "VoltageLossDetection":
        best_monitor, best_mode = "Validation VLD Balanced Accuracy", "max"
    else:
        best_monitor, best_mode = "Validation loss", "min"

    save_best_model_callback = SaveBestModelStateDict(
        monitor=best_monitor,
        mode=best_mode,
        filename="best_model_state_dict.pt",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="Validation loss",  # or whichever metric you track
        mode="min",
        save_last=True,
        save_top_k=0,
    )

    return [
        early_stop_callback,
        save_best_model_callback,
        checkpoint_callback,
        PhysicsGateMonitor(floor=0.10),  # receptive-field collapse alarm
    ]


def _num_nodes() -> int:
    """Node count for L.Trainer, or 1 when srun did not launch this process.
    +
      Lightning defaults num_nodes=1, so SLURMEnvironment.validate_settings
      rejects any multi-node allocation. The value must also be *correct*, not
      merely accepted: DDPStrategy.distributed_sampler_kwargs derives
      num_replicas from num_nodes * num_processes, so a wrong value silently
      mis-shards the dataset instead of raising.
    +
      Gated on SLURM_NTASKS, which is the same signal Lightning itself uses in
      lightning.fabric.plugins.environments.slurm._is_srun_used. SLURM_NNODES is
      also set inside an `salloc` shell, where the app may be started WITHOUT
      srun; trusting it there would make Lightning wait for ranks that never
      arrive. Outside SLURM this returns 1, i.e. exactly Lightning's default, so
      laptop, workstation and CI runs are unaffected.
    +"""
    if not os.environ.get("SLURM_NTASKS"):
        return 1
    try:
        return int(os.environ.get("SLURM_NNODES") or 1)
    except ValueError:
        return 1


def _global_rank() -> int:
    """This process's global rank, readable BEFORE torch.distributed exists.

    The MLflow logger has to be built before the Trainer, so neither
    trainer.global_rank nor torch.distributed.get_rank() is available yet.
    SLURM_PROCID is set per task by srun; RANK is the torchrun equivalent.
    """
    for var in ("SLURM_PROCID", "RANK"):
        value = os.environ.get(var)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def main_cli(args):
    if getattr(args, "tf32", False):
        torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+ GPUs

    # MLFlowLogger.__init__ eagerly constructs MlflowClient -> FileStore ->
    # _create_default_experiment(), and that path is NOT rank_zero_only. With
    # 16 ranks starting at once the losers die with
    # Exception: Yaml file './mlruns/.../0/meta.yaml' exists as ...
    # Lightning already guards every WRITE through the logger (`experiment` is
    # @rank_zero_experiment, log_metrics/log_hyperparams are @rank_zero_only),
    # so construction is the only unsafe part. Rather than make the race
    # narrower, give the store exactly one owner: rank 0. Non-zero ranks pass
    # logger=False, which disables logging without substituting a default one
    # (logger=None would create a TensorBoard/CSV logger instead).
    # Safe for checkpointing: ModelCheckpoint.setup resolves dirpath then calls
    # trainer.strategy.broadcast(dirpath), so every rank adopts rank 0's path
    # even though only rank 0 has a logger to derive it from.
    rank = _global_rank()
    logger = (
        MLFlowLogger(
            save_dir=args.log_dir,
            experiment_name=args.exp_name,
            run_name=args.run_name,
        )
        if rank == 0
        else False
    )

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    config_args = NestedNamespace(**base_config)

    L.seed_everything(config_args.seed, workers=True)

    normalizer_stats_path = getattr(args, "normalizer_stats", None)
    dataset_wrapper = getattr(args, "dataset_wrapper", None)
    dataset_wrapper_cache_dir = getattr(args, "dataset_wrapper_cache_dir", None)

    # CLI --num_workers overrides the YAML value (useful for debugging with 0)
    num_workers_override = getattr(args, "num_workers", None)
    if num_workers_override is not None:
        config_args.data.workers = num_workers_override

    _load_plugins(getattr(args, "plugins", []))
    _validate_dataset_wrapper(dataset_wrapper)

    litGrid = LitGridHeteroDataModule(
        config_args,
        args.data_path,
        normalizer_stats_path=normalizer_stats_path,
        dataset_wrapper=dataset_wrapper,
        dataset_wrapper_cache_dir=dataset_wrapper_cache_dir,
    )
    model = get_task(config_args, litGrid.data_normalizers)
    if args.command != "train":
        print(f"Loading model weights from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location="cpu")
        # Non-strict so that checkpoints saved before the VLDTopologyLoss EMA
        # buffers (loss_fn.*.ema_off_weight / ema_initialised) existed still
        # load; they then fall back to the per-batch decision threshold.
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(
                "Warning: checkpoint is missing "
                f"{len(load_result.missing_keys)} key(s): "
                f"{load_result.missing_keys}",
            )
        if load_result.unexpected_keys:
            print(
                "Warning: checkpoint has "
                f"{len(load_result.unexpected_keys)} unexpected key(s): "
                f"{load_result.unexpected_keys}",
            )
        model_keys = [
            k for k in load_result.missing_keys if k.startswith("model.")
        ]
        if model_keys:
            raise RuntimeError(
                "Checkpoint does not match the model architecture; missing "
                f"model weights: {model_keys}",
            )

    precision = "bf16-true" if getattr(args, "bfloat16", False) else None
    if precision:
        print("Using bfloat16 precision (via Lightning Trainer precision='bf16-true')")

    compile_mode = getattr(args, "compile", None)
    if compile_mode is not None:
        if compile_mode in ("max-autotune", "max-autotune-no-cudagraphs"):
            # Allow ATen GEMM as fallback so Triton configs that exceed GPU
            # shared-memory limits (e.g. triton_mm OOM) are skipped gracefully
            # instead of causing autotuning errors.
            import torch._inductor.config as inductor_cfg

            inductor_cfg.max_autotune_gemm_backends = "ATEN,TRITON"
        print(f"Compiling model with torch.compile(mode='{compile_mode}')")
        model.model = torch.compile(model.model, mode=compile_mode)

    trainer_kwargs = {}
    if precision:
        trainer_kwargs["precision"] = precision
    profiler = getattr(args, "profiler", None)

    trainer = L.Trainer(
        logger=logger,
        accelerator=config_args.training.accelerator,
        devices=config_args.training.devices,
        num_nodes=_num_nodes(),
        strategy=config_args.training.strategy,
        log_every_n_steps=1000,
        default_root_dir=args.log_dir,
        max_epochs=config_args.training.epochs,
        callbacks=get_training_callbacks(config_args),
        **trainer_kwargs,
        profiler=profiler,
    )
    if args.command == "train" or args.command == "finetune":
        trainer.fit(model=model, datamodule=litGrid)

    # The test phase MUST run on EVERY rank. VLDTask.on_test_end calls
    # dist.gather_object(self.test_outputs, gathered, dst=0), a collective over
    # the whole process group, and trainer.fit() does NOT tear that group down:
    # measured dist.is_initialized() is still True and get_world_size() still
    # returns the full world after fit() returns.

    # The test Trainer therefore mirrors the fit geometry rather than using
    # devices=1, num_nodes=1: on_test_end merges one shard per rank, so the test
    # set has to be sharded the same way it was during fit, and each rank stays
    # on the GPU it already owns instead of every rank piling onto device 0.

    if args.command != "predict":
        test_trainer = L.Trainer(
            logger=logger,
            accelerator=config_args.training.accelerator,
            devices=config_args.training.devices,  # fit geometry of training instead of explicit call
            num_nodes=_num_nodes(),  # fit geometry of training instead of explicit call
            strategy=config_args.training.strategy,  # fit geometry of training instead of explicit call
            log_every_n_steps=1,
            default_root_dir=args.log_dir,
            **trainer_kwargs,
            profiler=profiler,
        )
        test_trainer.test(model=model, datamodule=litGrid)

    # Only NOW is it safe to drop the non-zero ranks. Everything below reads
    # logger.save_dir/.experiment_id/.run_id, which is False.save_dir ->
    # AttributeError on a rank whose logger is disabled. No collectives remain
    # past this point, so leaving early here cannot strand rank 0.
    if rank != 0:
        return

    artifacts_dir = os.path.join(
        logger.save_dir,
        logger.experiment_id,
        logger.run_id,
        "artifacts",
    )

    compute_dc_ac = getattr(args, "compute_dc_ac_metrics", False)
    if compute_dc_ac:
        sn_mva = config_args.data.baseMVA
        for grid_name in config_args.data.networks:
            raw_dir = os.path.join(args.data_path, grid_name, "raw")
            print(f"\nComputing ground-truth AC/DC metrics for {grid_name}...")
            compute_ac_dc_metrics(artifacts_dir, raw_dir, grid_name, sn_mva)

    save_output = getattr(args, "save_output", False) or args.command == "predict"
    if save_output:
        if len(config_args.data.networks) > 1:
            raise NotImplementedError(
                "Predict/save_output with multiple grids is not yet supported.",
            )

        predict_trainer = L.Trainer(
            logger=logger,
            accelerator=config_args.training.accelerator,
            devices=1,
            num_nodes=1,
            log_every_n_steps=1,
            default_root_dir=args.log_dir,
            **trainer_kwargs,
            profiler=profiler,
        )
        predictions = predict_trainer.predict(model=model, datamodule=litGrid)

        rows = {key: [] for key in predictions[0].keys()}
        for batch in predictions:
            for key in rows:
                rows[key].append(batch[key])

        df = pd.DataFrame({key: np.concatenate(vals) for key, vals in rows.items()})

        grid_name = config_args.data.networks[0]
        if args.command == "predict":
            output_dir = args.output_path
        else:
            output_dir = os.path.join(artifacts_dir, "test")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{grid_name}_predictions.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved predictions to {out_path}")
