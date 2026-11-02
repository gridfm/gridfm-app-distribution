from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import json
import os
import shutil
import time
import torch


class EpochTimerCallback(Callback):
    """Records wall-clock duration and iteration rate of every training epoch."""

    def __init__(self):
        self.epoch_times: list[float] = []
        self._epoch_start: float | None = None
        self._batch_count: int = 0
        self._last_batch_count: int = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()
        self._batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch_count += 1

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start is not None:
            self.epoch_times.append(time.perf_counter() - self._epoch_start)
            self._last_batch_count = self._batch_count
            self._epoch_start = None

    @property
    def last_epoch_time(self) -> float | None:
        return self.epoch_times[-1] if self.epoch_times else None

    @property
    def last_epoch_iters_per_sec(self) -> float | None:
        t = self.last_epoch_time
        if t is None or t == 0 or self._last_batch_count == 0:
            return None
        return self._last_batch_count / t


VLD_TASK_NAME = "VoltageLossDetection"


def validation_monitors(args):
    """
    Validation metrics (name, mode) driving early stopping, the LR schedule and
    the exported best state dict.

    VoltageLossDetection defaults:
      early_stopping : "Validation VLD Balanced Accuracy", max - the quantity the
                       task is judged on; the total loss is dominated by the
                       MSE / PBE voltage terms and stops runs whose classifier is
                       still improving.
      lr_scheduler   : "Validation VLD Loss", min - the VLD objective alone. It is
                       smoother than balanced accuracy, which can sit flat for
                       many epochs before improving, so it does not decay the LR
                       during such plateaus as quickly.
      best_model     : "Validation VLD Balanced Accuracy", max.
    Other tasks keep "Validation loss", min, for all three.

    Overrides: callbacks.monitor / callbacks.monitor_mode (early stopping) and
    optimizer.lr_monitor / optimizer.lr_monitor_mode (LR schedule).
    """
    task_name = getattr(getattr(args, "task", None), "task_name", None)
    if task_name == VLD_TASK_NAME:
        early = ("Validation VLD Balanced Accuracy", "max")
        lr = ("Validation VLD Loss", "min")
        best = ("Validation VLD Balanced Accuracy", "max")
    else:
        early = lr = best = ("Validation loss", "min")
    callbacks_cfg = getattr(args, "callbacks", None)
    optimizer_cfg = getattr(args, "optimizer", None)
    early = (
        getattr(callbacks_cfg, "monitor", early[0]),
        getattr(callbacks_cfg, "monitor_mode", early[1]),
    )
    lr = (
        getattr(optimizer_cfg, "lr_monitor", lr[0]),
        getattr(optimizer_cfg, "lr_monitor_mode", lr[1]),
    )
    for name, (_, mode) in (("callbacks.monitor_mode", early), ("optimizer.lr_monitor_mode", lr)):
        if mode not in ("min", "max"):
            raise ValueError(f"{name} must be 'min' or 'max', got {mode!r}.")
    return {"early_stopping": early, "lr_scheduler": lr, "best_model": best}


class SaveBestModelStateDict(Callback):
    """Persist the best model state_dict according to a monitored validation metric.

    Records where and when the best state was saved (``best_model_path``,
    ``best_epoch``), so the test phase can evaluate exactly that state, and keeps
    this record in Lightning checkpoints, so a resumed run continues from the
    previous best instead of overwriting it with the first post-resume epoch.

    Next to the state dict a small ``<file>.json`` records its epoch / score /
    monitor. It is written together with the state dict, so it is the source of
    truth when a job was killed between saving the best state and writing the
    next Lightning checkpoint.
    """

    def __init__(
        self,
        monitor: str,
        mode: str = "min",
        filename: str = "best_model_state_dict.pt",
    ):
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.best_model_path = None
        self.best_epoch = None

    # ---------------------------------------------------------------- state
    def state_dict(self):
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_score": float(self.best_score),
            "best_model_path": self.best_model_path,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("monitor") != self.monitor or state_dict.get("mode") != self.mode:
            print(
                "SaveBestModelStateDict: checkpoint tracked "
                f"{state_dict.get('monitor')!r} ({state_dict.get('mode')}), this run tracks "
                f"{self.monitor!r} ({self.mode}); starting a new best.",
            )
            return
        self.best_score = float(state_dict["best_score"])
        self.best_model_path = state_dict.get("best_model_path")
        self.best_epoch = state_dict.get("best_epoch")

    @property
    def has_best(self):
        return self.best_score not in (float("inf"), -float("inf"))

    # ---------------------------------------------------------------- paths
    def _model_dir(self, trainer):
        logger = trainer.logger
        if isinstance(logger, MLFlowLogger):
            return os.path.join(
                logger.save_dir,
                logger.experiment_id,
                logger.run_id,
                "artifacts",
                "model",
            )
        if logger:
            return os.path.join(logger.save_dir, "model")
        return None

    def _current_path(self, trainer):
        model_dir = self._model_dir(trainer)
        return os.path.abspath(os.path.join(model_dir, self.filename)) if model_dir else None

    @staticmethod
    def _read_sidecar(path):
        try:
            with open(path + ".json") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _canonical_state_dict(pl_module):
        state_dict = pl_module.state_dict()
        return {
            key.replace("model._orig_mod.", "model."): value
            for key, value in state_dict.items()
        }

    # ---------------------------------------------------------------- resume
    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        """Reconcile the restored best with the files of the run being written."""
        current = self._current_path(trainer)

        # Restored best (from a resumed checkpoint) whose file is elsewhere.
        if self.best_model_path and current and self.best_model_path != current:
            if os.path.exists(self.best_model_path):
                same = os.path.exists(current) and os.path.samefile(self.best_model_path, current)
                if not same:
                    os.makedirs(os.path.dirname(current), exist_ok=True)
                    shutil.copy2(self.best_model_path, current)
                    if os.path.exists(self.best_model_path + ".json"):
                        shutil.copy2(self.best_model_path + ".json", current + ".json")
                    print(f"SaveBestModelStateDict: copied the previous best into this run: {current}")
                self.best_model_path = current
            elif os.path.exists(current):
                self.best_model_path = current
            else:
                print(
                    f"WARNING: SaveBestModelStateDict: the previous best ({self.monitor} = "
                    f"{self.best_score:.4f}, epoch {self.best_epoch}) was restored but its file "
                    f"{self.best_model_path} is missing; the test phase will use the final "
                    "weights unless a better epoch is saved.",
                )
                self.best_model_path = None

        # Resuming, and a best file already exists in this run but no state was
        # restored (a checkpoint written before this callback was stateful, or a
        # changed monitor): do not let the first epoch after the resume overwrite
        # it blindly. A fresh run keeps the original behaviour.
        resuming = bool(getattr(trainer, "ckpt_path", None))
        if resuming and current and os.path.exists(current) and self.best_model_path is None:
            sidecar = self._read_sidecar(current)
            if sidecar and sidecar.get("monitor") == self.monitor:
                self.best_score = float(sidecar["score"])
                self.best_epoch = sidecar.get("epoch")
                self.best_model_path = current
                print(
                    f"SaveBestModelStateDict: kept the existing best ({self.monitor} = "
                    f"{self.best_score:.4f}, epoch {self.best_epoch}).",
                )
            else:
                stem, ext = os.path.splitext(current)
                backup = f"{stem}.pre_resume{ext}"
                shutil.copy2(current, backup)
                print(
                    "WARNING: SaveBestModelStateDict: found a best state dict without a "
                    f"recorded score; kept a copy at {backup} before it is overwritten.",
                )

        # The sidecar is written with the state dict, so it wins if the job was
        # killed between the two (file of epoch N, checkpoint still saying M).
        if self.best_model_path and os.path.exists(self.best_model_path):
            sidecar = self._read_sidecar(self.best_model_path)
            if sidecar and sidecar.get("monitor") == self.monitor:
                self.best_score = float(sidecar["score"])
                self.best_epoch = sidecar.get("epoch")

    # ---------------------------------------------------------------- save
    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        if (self.mode == "min" and current < self.best_score) or (
            self.mode == "max" and current > self.best_score
        ):
            self.best_score = float(current)

            model_path = self._current_path(trainer)
            if model_path is None:
                return
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # Atomic write: a job killed mid-save never leaves a truncated file.
            tmp_path = model_path + ".tmp"
            torch.save(self._canonical_state_dict(pl_module), tmp_path)
            os.replace(tmp_path, model_path)
            self.best_model_path = model_path
            self.best_epoch = int(trainer.current_epoch)
            with open(model_path + ".json.tmp", "w") as f:
                json.dump(
                    {"epoch": self.best_epoch, "score": self.best_score, "monitor": self.monitor},
                    f,
                )
            os.replace(model_path + ".json.tmp", model_path + ".json")


class ConfigPatienceEarlyStopping(EarlyStopping):
    """EarlyStopping whose patience always comes from the config, also on resume
    (Lightning otherwise restores the checkpoint's patience)."""

    def load_state_dict(self, state_dict):
        patience = self.patience
        super().load_state_dict(state_dict)
        self.patience = patience
