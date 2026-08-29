import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
from gridfm_app_dist.io.registries import PHYSICS_DECODER_REGISTRY

from gridfm_app_dist.datasets.globals import (
    # Bus feature indices
    PD_H,
    QD_H,
    GS,
    BS,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    PG_OUT,
    QG_OUT,
    # Edge feature indices
    YFF_TT_R,
    YFF_TT_I,
    YFT_TF_R,
    YFT_TF_I,
)


class ComputeBranchFlow(nn.Module):
    """Compute sending-end branch flows (Pf, Qf) for all branches."""

    def forward(self, bus_data, edge_index, edge_attr):
        from_idx, to_idx = edge_index

        # Voltage magnitudes and angles
        Vf_mag, Vf_ang = bus_data[from_idx, VM_OUT], bus_data[from_idx, VA_OUT]
        Vt_mag, Vt_ang = bus_data[to_idx, VM_OUT], bus_data[to_idx, VA_OUT]

        # Real & imaginary voltage components
        Vf_r = Vf_mag * torch.cos(Vf_ang)
        Vf_i = Vf_mag * torch.sin(Vf_ang)
        Vt_r = Vt_mag * torch.cos(Vt_ang)
        Vt_i = Vt_mag * torch.sin(Vt_ang)

        # Branch admittance components
        Yfftt_r, Yfftt_i = edge_attr[:, YFF_TT_R], edge_attr[:, YFF_TT_I]
        Yfttf_r, Yfttf_i = edge_attr[:, YFT_TF_R], edge_attr[:, YFT_TF_I]

        # Sending-end currents
        Ift_r = Yfftt_r * Vf_r - Yfftt_i * Vf_i + Yfttf_r * Vt_r - Yfttf_i * Vt_i
        Ift_i = Yfftt_r * Vf_i + Yfftt_i * Vf_r + Yfttf_r * Vt_i + Yfttf_i * Vt_r

        # Sending-end power flows
        Pft = Vf_r * Ift_r + Vf_i * Ift_i
        Qft = Vf_i * Ift_r - Vf_r * Ift_i

        return Pft, Qft


class ComputeNodeInjection(nn.Module):
    """Aggregate branch flows into node-level incoming injections."""

    def forward(self, Pft, Qft, edge_index, num_bus):
        """
        Args:
            Pft, Qft: [num_edges] branch flows
            edge_index: [2, num_edges] (from_bus, to_bus)
            num_bus: number of bus nodes
        Returns:
            P_in, Q_in: aggregated incoming power per bus
        """
        from_idx, _ = edge_index  # Only sending end contributes as "incoming" to node
        P_in = scatter_add(Pft, from_idx, dim=0, dim_size=num_bus)
        Q_in = scatter_add(Qft, from_idx, dim=0, dim_size=num_bus)

        return P_in, Q_in


def compute_shunt_power(bus_data_pred, bus_data_orig):
    p_shunt = -bus_data_orig[:, GS] * bus_data_pred[:, VM_OUT] ** 2
    q_shunt = bus_data_orig[:, BS] * bus_data_pred[:, VM_OUT] ** 2
    return p_shunt, q_shunt


@PHYSICS_DECODER_REGISTRY.register("OptimalPowerFlow")
class PhysicsDecoderOPF(nn.Module):
    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig, agg_bus, mask_dict):
        mask_pv = mask_dict["PV"]
        mask_ref = mask_dict["REF"]

        mask_pvref = mask_pv | mask_ref

        # Shunt reactive power contribution
        _, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)

        # Reactive load
        Qd = bus_data_orig[:, QD_H]

        # ---- COMPUTE Qg FOR PV & REF ----
        # Nodal reactive balance:
        #     Qg = Q_in + Qd - q_shunt
        Qg_physics = Q_in + Qd - q_shunt

        # Use torch.where instead of boolean index-put to avoid aten.nonzero
        # (data-dependent shape) which causes inductor graph breaks under
        # torch.compile.
        Qg_new = torch.where(mask_pvref, Qg_physics, torch.zeros_like(Qg_physics))
        Pg_out = agg_bus  # Active generation (Pg)
        Qg_out = Qg_new  # Reactive gen (Qg)
        Vm_out = bus_data_pred[:, VM_OUT]  # Voltage magnitude
        Va_out = bus_data_pred[:, VA_OUT]  # Voltage angle

        # Concatenate into [num_buses, 4]
        output = torch.stack([Vm_out, Va_out, Pg_out, Qg_out], dim=1)

        return output


@PHYSICS_DECODER_REGISTRY.register("PowerFlow")
class PhysicsDecoderPF(nn.Module):
    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig, agg_bus, mask_dict):
        """
        PF decoder:
        - Compute Pg at REF bus
        - Compute Qg at PV + REF buses
        - PQ buses: Pg = 0, Qg = 0
        - Return stacked tensor [Vm, Va, Pg, Qg]
        """

        # Masks
        mask_pv = mask_dict["PV"]
        mask_ref = mask_dict["REF"]
        mask_pvref = mask_pv | mask_ref  # Qg computed here

        # --- Shunt contributions ---
        p_shunt, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)

        # --- Loads ---
        Pd = bus_data_orig[:, PD_H]
        Qd = bus_data_orig[:, QD_H]

        # ======================
        #   Qg (PV + REF)
        # ======================
        # Use torch.where instead of boolean index-put to avoid aten.nonzero
        # (data-dependent shape) which causes inductor graph breaks under
        # torch.compile.
        Qg_new = torch.where(mask_pvref, Q_in + Qd - q_shunt, torch.zeros_like(Q_in))

        # ======================
        #   Pg (REF only)
        # ======================
        Pg_ref = torch.where(mask_ref, P_in + Pd - p_shunt, torch.zeros_like(P_in))
        Pg_new = torch.where(mask_pv, agg_bus, Pg_ref)  # PV: keep predicted

        # Voltages
        Vm_out = bus_data_pred[:, VM_OUT]
        Va_out = bus_data_pred[:, VA_OUT]

        # Stack into [num_buses, 4] -> [Vm, Va, Pg, Qg]
        output = torch.stack([Vm_out, Va_out, Pg_new, Qg_new], dim=1)

        return output


@PHYSICS_DECODER_REGISTRY.register("StateEstimation")
class PhysicsDecoderSE(nn.Module):
    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig, agg_bus, mask_dict):
        p_shunt, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)
        Vm_out = bus_data_pred[:, VM_OUT]
        Va_out = bus_data_pred[:, VA_OUT]
        output = torch.stack([Vm_out, Va_out, P_in - p_shunt, Q_in - q_shunt], dim=1)
        return output

#########################
@PHYSICS_DECODER_REGISTRY.register("VoltageLossDetection")
class PhysicsDecoderVLD(nn.Module):
    """
    VLD decoder:
    Use the same physical decoding rule as PowerFlow for the bus outputs
    [Vm, Va, Pg, Qg]. The VLD-specific bus-status logit is produced by a
    separate model head and concatenated later in the model forward pass.
    """

    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig, agg_bus, mask_dict):
        mask_pv = mask_dict["PV"]
        mask_ref = mask_dict["REF"]
        mask_pvref = mask_pv | mask_ref

        p_shunt, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)

        Pd = bus_data_orig[:, PD_H]
        Qd = bus_data_orig[:, QD_H]

        Qg_new = torch.where(mask_pvref, Q_in + Qd - q_shunt, torch.zeros_like(Q_in))

        Pg_ref = torch.where(mask_ref, P_in + Pd - p_shunt, torch.zeros_like(P_in))
        Pg_new = torch.where(mask_pv, agg_bus, Pg_ref)

        Vm_out = bus_data_pred[:, VM_OUT]
        Va_out = bus_data_pred[:, VA_OUT]

        output = torch.stack([Vm_out, Va_out, Pg_new, Qg_new], dim=1)
        return output
########################

class ComputeNodeResiduals(nn.Module):
    """Compute net residuals per bus combining branch flows, generators, loads, and shunts."""

    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig):
        # Shunt contributions
        p_shunt, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)

        # Net residuals per bus
        residual_P = bus_data_pred[:, PG_OUT] - bus_data_orig[:, PD_H] + p_shunt - P_in
        residual_Q = bus_data_pred[:, QG_OUT] - bus_data_orig[:, QD_H] + q_shunt - Q_in

        return residual_P, residual_Q


def bound_with_sigmoid(pred, low, high):
    return low + (high - low) * torch.sigmoid(pred)


# ---------------------------------------------------------------------------
# Residual / correlation plotting helpers.
#
# NOTE (VLD integration): these three helpers are used by the base-task modules
# (pf_task.py / opf_task.py / se_task.py) for verbose evaluation plots. They were
# not part of the attached VLD files, so they are restored here verbatim from
# gridfm_graphkit/tasks/utils.py so those imports resolve. They are independent
# of the VoltageLossDetection task.
# ---------------------------------------------------------------------------


def residual_stats_by_type(residual, mask, bus_batch):
    """Return per-graph mean and max absolute residuals for a masked bus subset."""
    residual_masked = residual[mask]
    batch_masked = bus_batch[mask]
    abs_residual = torch.abs(residual_masked)

    # torch_scatter on MPS can dispatch into a CPU-only path for scatter_max.
    # Compute the grouped stats on CPU and move the results back so verbose
    # evaluation works without changing the torch/torch_scatter stack.
    if abs_residual.device.type == "mps":
        abs_residual_cpu = abs_residual.cpu()
        batch_masked_cpu = batch_masked.cpu()
        mean_res = scatter_mean(abs_residual_cpu, batch_masked_cpu, dim=0).to(
            abs_residual.device,
        )
        max_res, _ = scatter_max(abs_residual_cpu, batch_masked_cpu, dim=0)
        max_res = max_res.to(abs_residual.device)
    else:
        mean_res = scatter_mean(abs_residual, batch_masked, dim=0)
        max_res, _ = scatter_max(abs_residual, batch_masked, dim=0)
    return mean_res, max_res


def plot_residuals_histograms(outputs, dataset_name, plot_dir):
    """Plot mean/max residuals for P and Q, per bus type with consistent bins."""
    bus_types = ["REF", "PV", "PQ"]
    colors = ["#6baed6", "#fd8d3c", "#74c476"]  # PQ, PV, REF

    stats = [
        ("mean_residual_P", "Mean P Residual"),
        ("mean_residual_Q", "Mean Q Residual"),
        ("max_residual_P", "Max P Residual"),
        ("max_residual_Q", "Max Q Residual"),
    ]

    for stat_key, title in stats:
        # Gather all data first to compute common bin edges
        all_data = (
            torch.cat(
                [
                    torch.cat([d[f"{stat_key}_{bus_type}"] for d in outputs])
                    for bus_type in bus_types
                ],
            )
            .float()
            .numpy()
        )

        # Define bins across the entire data range
        bins = np.linspace(all_data.min(), all_data.max(), 61)

        plt.figure(figsize=(10, 6))
        for bus_type, color in zip(bus_types, colors):
            data = (
                torch.cat([d[f"{stat_key}_{bus_type}"] for d in outputs])
                .float()
                .numpy()
            )
            plt.hist(data, bins=bins, alpha=0.6, label=bus_type, color=color)

        plt.title(f"{title} per Bus Type in {dataset_name}")
        plt.xlabel("Residual (MW or MVar)")
        plt.ylabel("Frequency")
        plt.legend(title="Bus Type")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(plot_dir, f"{stat_key}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_correlation_by_node_type(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: dict,
    feature_labels: list,
    plot_dir: str,
    prefix: str = "",
    xlabel: str = "Target",
    ylabel: str = "Pred",
    qg_violation_mask: torch.Tensor = None,
):
    """Create correlation scatter plots per node type (PQ, PV, REF)."""
    os.makedirs(plot_dir, exist_ok=True)

    for node_type, mask in masks.items():
        if len(mask.shape) == 1:
            preds_masked = preds[mask]
            targets_masked = targets[mask]
        else:
            preds_masked = torch.where(mask, preds, 0)
            targets_masked = torch.where(mask, targets, 0)

        if preds_masked.numel() == 0:
            continue

        if qg_violation_mask is not None:
            qg_violation_mask_local = qg_violation_mask[mask].cpu().numpy()
        else:
            qg_violation_mask_local = None

        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        axes = axes.flatten()

        for i, (ax, label) in enumerate(zip(axes, feature_labels)):
            x = targets_masked[:, i].cpu().numpy().flatten()
            y = preds_masked[:, i].cpu().numpy().flatten()

            if label != "Qg" or qg_violation_mask_local is None:
                sns.scatterplot(x=x, y=y, s=6, alpha=0.4, ax=ax, edgecolor=None)
            else:
                normal_mask = ~qg_violation_mask_local
                viol_mask = qg_violation_mask_local

                sns.scatterplot(
                    x=x[normal_mask],
                    y=y[normal_mask],
                    s=6,
                    alpha=0.4,
                    ax=ax,
                    edgecolor=None,
                    label="Valid Qg",
                )
                sns.scatterplot(
                    x=x[viol_mask],
                    y=y[viol_mask],
                    s=8,
                    alpha=0.8,
                    ax=ax,
                    edgecolor="red",
                    color="red",
                    label="Qg violation",
                )
                ax.legend()

            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                linewidth=1.0,
                alpha=0.7,
            )

            corr = np.corrcoef(x, y)[0, 1]
            if label != "Qg" or qg_violation_mask_local is None:
                ax.set_title(f"{node_type} \u2013 {label}\nR\u00b2 = {corr**2:.3f}")
            else:
                num_violations = qg_violation_mask_local.sum().item()
                total_points = qg_violation_mask_local.shape[0]
                ax.set_title(
                    f"{node_type} \u2013 {label}\nR\u00b2 = {corr**2:.3f} - {num_violations} violations out of {total_points} predictions",
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        plt.tight_layout()
        filename = f"{prefix}_correlation_{node_type}.png"
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.close(fig)
