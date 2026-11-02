"""
Resume and best-checkpoint helpers for the train / finetune entry point.

Kept free of data / model imports so they can be unit-tested on their own.
"""

import glob
import os

import torch

from gridfm_app_dist.training.callbacks import SaveBestModelStateDict


def run_id_from_checkpoint(ckpt_path):
    """MLflow run id of a ``<save_dir>/<experiment_id>/<run_id>/checkpoints/*.ckpt`` path."""
    if not ckpt_path:
        return None
    parent = os.path.dirname(os.path.abspath(ckpt_path))
    if os.path.basename(parent) != "checkpoints":
        return None
    return os.path.basename(os.path.dirname(parent)) or None


def resolve_resume(log_dir, resume_ckpt, resume_run_id=None):
    """
    Validate a resume request and return the MLflow run id to continue (or None).

    Runs identically on every rank (shared file system), so a bad request fails
    everywhere at once instead of leaving the other ranks waiting at a barrier.
    """
    if resume_run_id and not resume_ckpt:
        print("WARNING: --resume_run_id is ignored without --resume_from_checkpoint.")
        return None
    if not resume_ckpt:
        return None
    if not os.path.isfile(resume_ckpt):
        raise FileNotFoundError(f"--resume_from_checkpoint {resume_ckpt} does not exist")
    run_id = resume_run_id or run_id_from_checkpoint(resume_ckpt)
    if run_id is None:
        print(
            "WARNING: could not tell which MLflow run the checkpoint belongs to; "
            "the resumed training is logged into a NEW run (use --resume_run_id).",
        )
        return None
    if not glob.glob(os.path.join(log_dir, "*", run_id, "meta.yaml")):
        raise ValueError(
            f"MLflow run {run_id!r} (from "
            f"{'--resume_run_id' if resume_run_id else 'the checkpoint path'}) "
            f"was not found under {log_dir}. Pass --resume_run_id, or the right --log_dir.",
        )
    return run_id


def _global_rank():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def broadcast_from_rank0(obj):
    """Rank 0's ``obj`` on every rank (identity when not distributed)."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]
    return obj


def load_best_state_for_test(model, callbacks):
    """
    Load the state the SaveBestModelStateDict callback exported, so the test phase
    evaluates the same weights as ``best_model_state_dict.pt`` (not the last epoch).
    Must run on every rank: rank 0 knows the path and broadcasts it, so every
    rank takes the same branch.
    """
    best = next((cb for cb in callbacks if isinstance(cb, SaveBestModelStateDict)), None)
    info = None
    # Only rank 0 runs the callback's save / resume reconciliation, so only its
    # record is authoritative; the others take it from the broadcast.
    if best is not None and _global_rank() == 0:
        if best.best_model_path and os.path.exists(best.best_model_path):
            info = (best.best_model_path, best.best_epoch, float(best.best_score), best.monitor)
        elif best.has_best:
            print(
                "WARNING: a best score was recorded but its state dict file is missing; "
                "the test phase falls back to the final weights.",
            )
    info = broadcast_from_rank0(info)
    if info is None:
        print("Test phase: no best state dict was saved; testing the final weights.")
        return None
    path, epoch, score, monitor = info
    state_dict = torch.load(path, map_location="cpu")
    # The file holds canonical keys ("model.*"); a torch.compile'd model expects
    # "model._orig_mod.*".
    if any(k.startswith("model._orig_mod.") for k in model.state_dict()):
        state_dict = {
            (
                "model._orig_mod." + k[len("model.") :]
                if k.startswith("model.") and not k.startswith("model._orig_mod.")
                else k
            ): v
            for k, v in state_dict.items()
        }
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if k.startswith("model.")]
    if missing:
        raise RuntimeError(f"Best state dict {path} does not match the model: missing {missing[:5]}")
    print(f"Test phase: testing the best state (epoch {epoch}, {monitor} = {score:.4f}) from {path}")
    return info
