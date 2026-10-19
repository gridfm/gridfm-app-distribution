import os
import torch
import torch.distributed as dist
import pandas as pd
from lightning.pytorch.loggers import MLFlowLogger

from gridfm_app_dist.io.registries import TASK_REGISTRY
from gridfm_app_dist.tasks.reconstruction_tasks import ReconstructionTask
from gridfm_app_dist.training.loss import VLDTopologyLoss
from gridfm_app_dist.io.vld_evaluation import evaluate_vld_predictions
from gridfm_app_dist.datasets.globals import (
    VM_OUT,
    VM_H,
    BUS_STATUS_TARGET,
    BUS_STATUS_LOGIT_OUT,
)


@TASK_REGISTRY.register("VoltageLossDetection")
class VoltageLossDetectionTask(ReconstructionTask):
    """
    Topology-aware voltage loss detection task.

    Uses the standard ReconstructionTask training/validation flow and adds
    VLD-specific test/predict metrics for bus status and Vm behavior.
    """

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)

    def _decision_threshold(self, loss_dict):
        """
        Threshold on the status probability above which a bus is called energised.
        The loss is trained with the de-energised class up-weighted by w_off,
        which moves its optimum away from the posterior, so the matching threshold is 1 / (1 + w_off)

        Preference order:
          1. the training-time w_off stored in the VLDTopologyLoss EMA buffer (restored from the checkpoint with the rest of the state dict);
          2. the per-batch value the loss just computed ("VLD Decision Threshold"), used when no EMA is available;
          3. 0.5, if the configured losses contain no VLDTopologyLoss at all.
        """
        for loss_fn in getattr(self.loss_fn, "loss_functions", []):
            if isinstance(loss_fn, VLDTopologyLoss):
                theta = loss_fn.decision_threshold()
                if theta is not None:
                    return float(theta)
                break
        theta = loss_dict.get("VLD Decision Threshold")
        if theta is not None:
            return float(theta)
        return 0.5

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output, loss_dict = self.shared_step(batch)
        dataset_name = self.args.data.networks[dataloader_idx]

        bus_pred = output["bus"]
        bus_target = batch.y_dict["bus"]

        threshold = self._decision_threshold(loss_dict)
        status_prob = torch.sigmoid(bus_pred[:, BUS_STATUS_LOGIT_OUT])
        status_pred = (status_prob >= threshold).float()
        status_true = bus_target[:, BUS_STATUS_TARGET].float()

        vm_pred = bus_pred[:, VM_OUT]
        vm_true = bus_target[:, VM_H]

        # Classification metrics (accuracy, off recall / precision / F1, on recall, balanced accuracy) are already in loss_dict,
        # computed by the loss at the same threshold. Only the voltage metrics are added here.
        off_mask = status_true < 0.5
        on_mask = status_true >= 0.5

        off_vm_mae = (
            vm_pred[off_mask].abs().mean()
            if off_mask.any()
            else torch.tensor(0.0, device=vm_pred.device)
        )
        on_vm_rmse = (
            torch.sqrt(torch.mean((vm_pred[on_mask] - vm_true[on_mask]) ** 2))
            if on_mask.any()
            else torch.tensor(0.0, device=vm_pred.device)
        )

        loss_dict["OFF Vm MAE"] = off_vm_mae.detach()
        loss_dict["ON Vm RMSE"] = on_vm_rmse.detach()
        loss_dict["Test loss"] = loss_dict.pop("loss").detach()

        for metric, value in loss_dict.items():
            metric_name = f"{dataset_name}/{metric}"
            self.log(
                metric_name,
                value,
                batch_size=batch.num_graphs,
                add_dataloader_idx=False,
                sync_dist=True,
                logger=False,
            )

        bus_batch = batch.batch_dict["bus"]
        scenario_ids = batch["scenario_id"][bus_batch]
        local_bus_idx = torch.cat(
            [
                torch.arange(c, device=bus_batch.device)
                for c in torch.bincount(bus_batch)
            ],
        )

        self.test_outputs[dataloader_idx].append(
            {
                "dataset": dataset_name,
                "scenario": scenario_ids.detach().cpu(),
                "bus": local_bus_idx.detach().cpu(),
                "status_prob": status_prob.detach().cpu(),
                "status_pred": status_pred.detach().cpu(),
                "status_true": status_true.detach().cpu(),
                "vm_pred": vm_pred.detach().cpu(),
                "vm_true": vm_true.detach().cpu(),
                "threshold": torch.full_like(status_prob, threshold).cpu(),
            },
        )

    def on_test_end(self):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [None] * world_size if dist.get_rank() == 0 else None
            dist.gather_object(self.test_outputs, gathered, dst=0)
            if dist.get_rank() == 0:
                merged = {i: [] for i in range(len(self.args.data.networks))}
                for rank_data in gathered:
                    for dl_idx, batches in rank_data.items():
                        merged[dl_idx].extend(batches)
                self.test_outputs = merged

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        if isinstance(self.logger, MLFlowLogger):
            artifact_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
            )
        else:
            artifact_dir = self.logger.save_dir

        test_dir = os.path.join(artifact_dir, "test")
        os.makedirs(test_dir, exist_ok=True)

        for dataset_idx, outputs in self.test_outputs.items():
            if not outputs:
                continue

            dataset_name = self.args.data.networks[dataset_idx]
            scenario = torch.cat([o["scenario"] for o in outputs]).numpy()
            bus = torch.cat([o["bus"] for o in outputs]).numpy()
            status_prob = torch.cat([o["status_prob"] for o in outputs]).numpy()
            status_pred = torch.cat([o["status_pred"] for o in outputs]).numpy()
            status_true = torch.cat([o["status_true"] for o in outputs]).numpy()
            vm_pred = torch.cat([o["vm_pred"] for o in outputs]).numpy()
            vm_true = torch.cat([o["vm_true"] for o in outputs]).numpy()
            threshold = torch.cat([o["threshold"] for o in outputs]).numpy()

            df = pd.DataFrame(
                {
                    "scenario": scenario,
                    "bus": bus,
                    "status_prob": status_prob,
                    "status_pred": status_pred,
                    "status_true": status_true,
                    "threshold": threshold,
                    "vm_pred": vm_pred,
                    "vm_true": vm_true,
                },
            )
            df.to_csv(
                os.path.join(test_dir, f"{dataset_name}_vld_predictions.csv"),
                index=False,
            )
            self._run_evaluation(df, os.path.join(test_dir, f"{dataset_name}_eval"))

        self.test_outputs.clear()

    def _run_evaluation(self, df, out_dir):
        """
        Metrics + figures for the model's predictions in ``df`` at the
        threshold the task used (see ``io.vld_evaluation``). Metric names
        mirror the ones VLDTopologyLoss logs. Settings come from an optional
        ``evaluation:`` block in the config (enabled, n_boot, seed); all have
        defaults.
        """
        ev = getattr(self.args, "evaluation", None)
        if ev is not None and not getattr(ev, "enabled", True):
            return None
        kwargs = {}
        if ev is not None:
            for name in ("n_boot", "seed"):
                if getattr(ev, name, None) is not None:
                    kwargs[name] = getattr(ev, name)
        try:
            metrics = evaluate_vld_predictions(
                df,
                out_dir,
                model_threshold=float(df["threshold"].iloc[0]),
                **kwargs,
            )
        except Exception as exc:  # evaluation must never break a test run
            print(f"[VLD evaluation] skipped: {exc!r}")
            return None

        if isinstance(self.logger, MLFlowLogger):
            scalar = {
                f"eval/{k}": float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and v == v
            }
            self.logger.log_metrics(scalar)
            self.logger.experiment.log_artifacts(self.logger.run_id, out_dir)
        print(f"[VLD evaluation] figures and metrics written to {out_dir}")
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output, loss_dict = self.shared_step(batch)

        bus_pred = output["bus"]
        threshold = self._decision_threshold(loss_dict)
        status_prob = torch.sigmoid(bus_pred[:, BUS_STATUS_LOGIT_OUT])

        bus_batch = batch.batch_dict["bus"]
        scenario_ids = batch["scenario_id"][bus_batch]

        local_bus_idx = torch.cat(
            [
                torch.arange(c, device=bus_batch.device)
                for c in torch.bincount(bus_batch)
            ],
        )

        return {
            "scenario": scenario_ids.cpu().numpy(),
            "bus": local_bus_idx.cpu().numpy(),
            "vm_pred": bus_pred[:, VM_OUT].detach().cpu().numpy(),
            "vm_true": batch.y_dict["bus"][:, VM_H].detach().cpu().numpy(),
            "status_prob": status_prob.detach().cpu().numpy(),
            "status_pred": (status_prob >= threshold).float().detach().cpu().numpy(),
            "threshold": torch.full_like(status_prob, threshold).detach().cpu().numpy(),
            "status_true": batch.y_dict["bus"][:, BUS_STATUS_TARGET]
            .detach()
            .cpu()
            .numpy(),
        }
