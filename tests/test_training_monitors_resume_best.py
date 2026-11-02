"""D1 (validation monitors), D2 (resume) and E1 (test the best state) helpers."""

import json
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("pytorch_lightning")

from torch import nn  # noqa: E402

from gridfm_app_dist.io.param_handler import NestedNamespace  # noqa: E402
from gridfm_app_dist.training.callbacks import (  # noqa: E402
    SaveBestModelStateDict,
    validation_monitors,
)
from gridfm_app_dist.training.checkpointing import (  # noqa: E402
    load_best_state_for_test,
    resolve_resume,
    run_id_from_checkpoint,
)

BA = "Validation VLD Balanced Accuracy"


def _args(task, callbacks=None, optimizer=None):
    return NestedNamespace(
        task={"task_name": task},
        callbacks=callbacks or {"patience": 100, "tol": 0},
        optimizer=optimizer
        or {"learning_rate": 1e-4, "lr_decay": 0.7, "lr_patience": 10},
    )


# ------------------------------------------------------------------ D1
def test_vld_monitors_default_to_classification_metrics():
    m = validation_monitors(_args("VoltageLossDetection"))
    assert m["early_stopping"] == (BA, "max")
    assert m["lr_scheduler"] == ("Validation VLD Loss", "min")
    assert m["best_model"] == (BA, "max")


def test_other_tasks_keep_the_validation_loss():
    m = validation_monitors(_args("PowerFlow"))
    assert set(m.values()) == {("Validation loss", "min")}


def test_monitor_overrides_and_validation():
    args = _args(
        "VoltageLossDetection",
        callbacks={
            "patience": 100,
            "tol": 0,
            "monitor": "Validation loss",
            "monitor_mode": "min",
        },
        optimizer={"lr_monitor": BA, "lr_monitor_mode": "max"},
    )
    m = validation_monitors(args)
    assert m["early_stopping"] == ("Validation loss", "min")
    assert m["lr_scheduler"] == (BA, "max")
    with pytest.raises(ValueError):
        validation_monitors(
            _args("VoltageLossDetection", callbacks={"monitor_mode": "up"}),
        )


def test_resumed_lr_schedule_restarts_when_the_metric_changed():
    from gridfm_app_dist.tasks.base_task import BaseTask

    task = SimpleNamespace(args=_args("VoltageLossDetection"))
    plateau = {
        "mode": "min",
        "best": 0.2,
        "mode_worse": float("inf"),
        "num_bad_epochs": 7,
        "patience": 3,
        "factor": 0.5,
    }
    ckpt = {
        "lr_schedulers": [dict(plateau)],
    }  # older checkpoint: no lr_monitor recorded
    BaseTask.on_load_checkpoint(task, ckpt)
    state = ckpt["lr_schedulers"][0]
    assert (state["best"], state["num_bad_epochs"]) == (float("inf"), 0)
    assert (state["patience"], state["factor"]) == (10, 0.7)  # from the config
    ckpt = {
        "lr_schedulers": [dict(plateau)],
        "lr_monitor": ["Validation VLD Loss", "min"],
    }
    BaseTask.on_load_checkpoint(task, ckpt)
    assert ckpt["lr_schedulers"][0]["best"] == 0.2  # same metric: tracking continues


# ------------------------------------------------------------------ D2
def test_run_id_is_read_from_the_checkpoint_path():
    assert run_id_from_checkpoint("mlruns/1234/abcd/checkpoints/last.ckpt") == "abcd"
    assert run_id_from_checkpoint("somewhere/last.ckpt") is None
    assert run_id_from_checkpoint(None) is None


def test_resolve_resume_validates_the_run(tmp_path):
    run = tmp_path / "mlruns" / "123" / "abcd"
    (run / "checkpoints").mkdir(parents=True)
    (run / "meta.yaml").write_text("run_id: abcd\n")
    ckpt = run / "checkpoints" / "last.ckpt"
    ckpt.write_bytes(b"x")
    log_dir = str(tmp_path / "mlruns")
    assert resolve_resume(log_dir, str(ckpt)) == "abcd"
    assert resolve_resume(log_dir, None) is None
    with pytest.raises(FileNotFoundError):
        resolve_resume(log_dir, str(tmp_path / "nope.ckpt"))
    with pytest.raises(ValueError):
        resolve_resume(log_dir, str(ckpt), resume_run_id="not-a-run")


def _trainer(save_dir, metric=None, epoch=0, ckpt_path=None):
    return SimpleNamespace(
        ckpt_path=ckpt_path,
        logger=SimpleNamespace(save_dir=str(save_dir)),
        sanity_checking=False,
        current_epoch=epoch,
        callback_metrics={} if metric is None else {BA: torch.tensor(metric)},
    )


def test_best_state_survives_a_resume():
    cb = SaveBestModelStateDict(BA, mode="max")
    cb.best_score, cb.best_model_path, cb.best_epoch = 0.93, "/runs/1/model.pt", 172
    resumed = SaveBestModelStateDict(BA, mode="max")
    resumed.load_state_dict(cb.state_dict())
    assert (resumed.best_score, resumed.best_model_path, resumed.best_epoch) == (
        0.93,
        "/runs/1/model.pt",
        172,
    )
    other = SaveBestModelStateDict("Validation loss", mode="min")
    other.load_state_dict(cb.state_dict())
    assert other.best_score == float("inf") and other.best_model_path is None


def test_best_file_is_written_with_its_sidecar(tmp_path):
    cb = SaveBestModelStateDict(BA, mode="max")
    module = nn.Linear(2, 2)
    cb.on_validation_end(_trainer(tmp_path, 0.8, epoch=3), module)
    path = os.path.join(str(tmp_path), "model", "best_model_state_dict.pt")
    assert cb.best_model_path == os.path.abspath(path) and cb.best_epoch == 3
    assert json.load(open(path + ".json")) == {
        "epoch": 3,
        "score": pytest.approx(0.8),
        "monitor": BA,
    }
    assert not os.path.exists(path + ".tmp")


def test_existing_best_is_not_overwritten_after_a_stateless_resume(tmp_path):
    first = SaveBestModelStateDict(BA, mode="max")
    module = nn.Linear(2, 2)
    first.on_validation_end(_trainer(tmp_path, 0.9, epoch=5), module)
    saved = torch.load(first.best_model_path)

    resumed = SaveBestModelStateDict(BA, mode="max")  # no state restored
    resumed.on_fit_start(_trainer(tmp_path, ckpt_path="last.ckpt"), module)
    assert resumed.best_score == pytest.approx(0.9) and resumed.best_epoch == 5
    with torch.no_grad():
        module.weight.fill_(7.0)
    resumed.on_validation_end(_trainer(tmp_path, 0.6, epoch=6), module)  # worse
    assert torch.equal(torch.load(first.best_model_path)["weight"], saved["weight"])


def test_fresh_run_does_not_adopt_a_previous_best(tmp_path):
    first = SaveBestModelStateDict(BA, mode="max")
    module = nn.Linear(2, 2)
    first.on_validation_end(_trainer(tmp_path, 0.9, epoch=5), module)
    fresh = SaveBestModelStateDict(BA, mode="max")
    fresh.on_fit_start(_trainer(tmp_path), module)  # not resuming
    assert not fresh.has_best and fresh.best_model_path is None


def test_sidecar_wins_over_a_stale_checkpoint_record(tmp_path):
    cb = SaveBestModelStateDict(BA, mode="max")
    module = nn.Linear(2, 2)
    cb.on_validation_end(_trainer(tmp_path, 0.85, epoch=6), module)
    stale = SaveBestModelStateDict(BA, mode="max")
    stale.load_state_dict(
        {
            "monitor": BA,
            "mode": "max",
            "best_score": 0.8,
            "best_model_path": cb.best_model_path,
            "best_epoch": 5,
        },
    )
    stale.on_fit_start(_trainer(tmp_path, ckpt_path="last.ckpt"), module)
    assert (stale.best_epoch, stale.best_score) == (6, pytest.approx(0.85))


# ------------------------------------------------------------------ E1
class _Task(nn.Module):
    def __init__(self, compiled=False):
        super().__init__()
        inner = nn.Linear(2, 2)
        if compiled:  # mimic torch.compile's OptimizedModule key layout
            wrapper = nn.Module()
            wrapper._orig_mod = inner
            self.model = wrapper
        else:
            self.model = inner


@pytest.mark.parametrize("compiled", [False, True])
def test_test_phase_loads_the_best_state(tmp_path, compiled):
    best = _Task()
    with torch.no_grad():
        best.model.weight.fill_(3.0)
    path = tmp_path / "best_model_state_dict.pt"
    torch.save(best.state_dict(), path)  # canonical "model.*" keys

    cb = SaveBestModelStateDict(BA, mode="max")
    cb.best_model_path, cb.best_epoch, cb.best_score = str(path), 7, 0.9

    task = _Task(compiled=compiled)
    assert load_best_state_for_test(task, [cb])[1] == 7
    weight = task.model._orig_mod.weight if compiled else task.model.weight
    assert torch.all(weight == 3.0)


def test_test_phase_falls_back_to_final_weights_without_a_best_file():
    task = _Task()
    before = task.model.weight.clone()
    assert (
        load_best_state_for_test(task, [SaveBestModelStateDict("Validation loss")])
        is None
    )
    assert torch.equal(task.model.weight, before)
