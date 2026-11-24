import os
import time
from abc import ABC, abstractmethod
import lightning as L
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseTask(L.LightningModule, ABC):
    """
    Abstract base LightningModule for feature reconstruction tasks.
    Contains shared training/validation/test logic, logging, and optimizer setup.
    """

    def __init__(self, args, data_normalizers):
        super().__init__()
        self.args = args
        self.data_normalizers = data_normalizers
        self.save_hyperparameters()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if getattr(device, "type", None) == "mps" and hasattr(batch, "stores"):
            for store in batch.stores:
                for key, val in store.items():
                    if isinstance(val, torch.Tensor) and val.dtype == torch.float64:
                        store[key] = val.to(torch.float32)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        if not hasattr(self, "model"):
            return batch
        try:
            target_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            return batch
        if target_dtype == torch.float32:
            return batch
        for store in batch.stores:
            for key, val in store.items():
                if isinstance(val, torch.Tensor) and val.is_floating_point():
                    store[key] = val.to(target_dtype)
        return batch

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass"""
        pass

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Non-finite guard (plan item 1.6)
    # ------------------------------------------------------------------
    # A single NaN/Inf loss or gradient used to reach AdamW and poison every
    # weight; the best-model export then saved a non-finite state. The guard:
    #   1. training_step: a non-finite loss is replaced by 0 * sum(params), so
    #      backward still runs on every rank (DDP stays in step) but contributes
    #      zero gradient; the step is flagged.
    #   2. on_before_optimizer_step (after the DDP all-reduce, before clipping):
    #      the flag and the finiteness of the gradient norm are summed over the
    #      ranks, so every rank takes the same decision. A bad step sets all
    #      gradients to None, which makes AdamW skip the update entirely (no
    #      weight decay, no moment update).
    #   3. After `training.max_nonfinite_steps` consecutive bad steps (default
    #      20) every rank raises, instead of training on silently.
    # The pre-clipping gradient norm is logged as "Gradient Norm".

    def guard_non_finite_loss(self, loss):
        if torch.isfinite(loss).all():
            self._nonfinite_loss = False
            return loss
        self._nonfinite_loss = True
        return sum(
            (p.sum() * 0.0 for p in self.parameters() if p.requires_grad),
            loss.new_zeros(()),
        )

    def on_before_optimizer_step(self, optimizer):
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if grads:
            grad_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads]),
            )
        else:
            grad_norm = torch.zeros((), device=self.device)
        bad = torch.tensor(
            float(getattr(self, "_nonfinite_loss", False) or not torch.isfinite(grad_norm)),
            device=grad_norm.device,
        )
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.world_size > 1:
            bad = trainer.strategy.reduce(bad, reduce_op="sum")
        self._nonfinite_loss = False
        if float(bad) > 0:
            for p in self.parameters():
                p.grad = None
            self._nonfinite_streak = getattr(self, "_nonfinite_streak", 0) + 1
            self._nonfinite_total = getattr(self, "_nonfinite_total", 0) + 1
            limit = int(getattr(self.args.training, "max_nonfinite_steps", 20))
            if self.global_rank == 0:
                print(
                    f"[non-finite guard] step {self.global_step}: non-finite loss or gradient, "
                    f"update skipped ({self._nonfinite_streak} in a row, "
                    f"{self._nonfinite_total} in total).",
                )
            if self._nonfinite_streak >= limit:
                raise RuntimeError(
                    f"{self._nonfinite_streak} consecutive optimizer steps had a non-finite "
                    f"loss or gradient (training.max_nonfinite_steps = {limit}). Stopping; "
                    "the last finite best_model_state_dict.pt is unaffected.",
                )
        else:
            self._nonfinite_streak = 0
            self.log(
                "Gradient Norm",
                grad_norm,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
            )
        self.log(
            "Non-finite Steps Skipped",
            float(getattr(self, "_nonfinite_total", 0)),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @rank_zero_only
    def on_fit_start(self):
        if isinstance(self.logger, MLFlowLogger):
            log_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
                "stats",
            )
        else:
            log_dir = os.path.join(self.logger.save_dir, "stats")

        os.makedirs(log_dir, exist_ok=True)

        log_stats_path = os.path.join(log_dir, "normalization_stats.txt")
        with open(log_stats_path, "w") as log_file:
            for i, normalizer in enumerate(self.data_normalizers):
                log_file.write(
                    f"Data Normalizer {self.args.data.networks[i]} stats:\n{normalizer.get_stats()}\n\n",
                )

        stats_dict = {}
        for i, normalizer in enumerate(self.data_normalizers):
            stats_dict[self.args.data.networks[i]] = normalizer.get_stats()
        torch.save(stats_dict, os.path.join(log_dir, "normalizer_stats.pt"))

    def on_save_checkpoint(self, checkpoint):
        # Record which metric the LR schedule follows, so a resume can tell
        # whether the checkpoint's ReduceLROnPlateau state is comparable.
        from gridfm_app_dist.training.callbacks import validation_monitors

        checkpoint["lr_monitor"] = list(validation_monitors(self.args)["lr_scheduler"])

    def on_load_checkpoint(self, checkpoint):
        """
        Align the restored ReduceLROnPlateau state with the current config.

        Lightning restores the scheduler's whole state after this hook, including
        mode, best value, patience and factor. If the checkpoint followed a
        different metric (or none was recorded: an older checkpoint), its "best"
        is not comparable with the new metric, so the plateau tracking restarts.
        patience / factor always come from the config.
        """
        from gridfm_app_dist.training.callbacks import validation_monitors

        name, mode = validation_monitors(self.args)["lr_scheduler"]
        same_metric = checkpoint.get("lr_monitor") == [name, mode]
        for state in checkpoint.get("lr_schedulers", []) or []:
            if not isinstance(state, dict) or "num_bad_epochs" not in state:
                continue  # not a ReduceLROnPlateau state
            state["patience"] = self.args.optimizer.lr_patience
            state["factor"] = self.args.optimizer.lr_decay
            if not same_metric:
                worse = float("inf") if mode == "min" else -float("inf")
                state.update(mode=mode, mode_worse=worse, best=worse, num_bad_epochs=0)
                print(
                    f"LR schedule: checkpoint followed {checkpoint.get('lr_monitor')}, "
                    f"now {[name, mode]}; plateau tracking restarted (current LR kept).",
                )

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.optimizer.learning_rate,
            betas=(self.args.optimizer.beta1, self.args.optimizer.beta2),
        )
        from gridfm_app_dist.training.callbacks import validation_monitors

        lr_monitor, lr_mode = validation_monitors(self.args)["lr_scheduler"]
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=lr_mode,
            factor=self.args.optimizer.lr_decay,
            patience=self.args.optimizer.lr_patience,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": lr_monitor,
                "reduce_on_plateau": True,
            },
        }
