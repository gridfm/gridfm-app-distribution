import torch.nn.functional as F
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from gridfm_dist_app.io.registries import LOSS_REGISTRY
from torch_scatter import scatter_add
from torch_geometric.utils import to_torch_coo_tensor

from gridfm_dist_app.datasets.globals import (
    # Bus feature indices
    QG_H,
    VM_H,
    VA_H,
    QD_H,
    PD_H,
    GS,
    BS,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    QG_OUT,
    PG_OUT,
    PD_OUT,
    QD_OUT,
    BUS_STATUS_TARGET,
    BUS_STATUS_LOGIT_OUT,
    # Edge feature indices
    YFF_TT_R,
    YFF_TT_I,
    YFT_TF_R,
    YFT_TF_I,
    # Qg limits
    MIN_QG_H,
    MAX_QG_H,
    # Generator feature indices
    PG_H,
    #####################
    ## Indices of features of the VLD task
    # Bus feature indices
    BUS_BASE_STATUS_H,
    BUS_CONT_H,
    B_ON,
    # Branch feature indices
    BRANCH_BASE_STATUS_E,
    BRANCH_CONT_E,
    #####################
)


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all custom loss functions.
    """

    @abstractmethod
    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        """
        Compute the loss.

        Parameters:
        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.
        - model: Optional model reference for accessing internal states.

        Returns:
        - A dictionary with the total loss and any additional metrics.
        """
        pass


@LOSS_REGISTRY.register("MaskedMSE")
class MaskedMSELoss(BaseLoss):
    """
    Mean Squared Error loss computed only on masked elements.

    Heterogeneous convention: ``pred``/``target``/``mask`` are the per-node-type
    dicts produced by the model, ``batch.y_dict`` and ``batch.mask_dict``.

    The three bus tensors use three DIFFERENT column layouts, so they can only be
    compared through an explicit index map (indexing them with a single mask is
    what raised ``TypeError: unhashable type: 'dict'`` / shape errors before):

      pred_dict["bus"]  : [Vm, Va, Pg, Qg] (+ status_logit for VLD)  -> *_OUT indices
      target_dict["bus"]: [Pd, Qd, Qg, Vm, Va, bus_status_target, Pg] -> *_H indices
      mask_dict["bus"]  : the 17-wide bus INPUT layout                -> *_H indices

    The label and mask layouts agree only on columns 0..4 (Pd, Qd, Qg, Vm, Va);
    beyond that they diverge (column 6 is PG_BUS_H in the label but PV_H in the
    input/mask layout). Only the reconstruction channels inside that shared
    region are used, so widening the bus label cannot silently misalign the loss.
    """

    # (prediction column, label/mask column) for the masked reconstruction channels.
    BUS_COLS = ((VM_OUT, VM_H), (VA_OUT, VA_H), (QG_OUT, QG_H))

    def __init__(self, loss_args, args):
        super(MaskedMSELoss, self).__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        if not isinstance(pred, dict):
            loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
            return {"loss": loss, "Masked MSE loss": loss.detach()}

        pred_bus, target_bus, mask_bus = pred["bus"], target["bus"], mask["bus"]
        selected_pred = [pred_bus[:, p][mask_bus[:, t]] for p, t in self.BUS_COLS]
        selected_target = [target_bus[:, t][mask_bus[:, t]] for _, t in self.BUS_COLS]

        gen_mask = mask["gen"][:, : (PG_H + 1)]
        selected_pred.append(pred["gen"][:, : (PG_H + 1)][gen_mask])
        selected_target.append(target["gen"][:, : (PG_H + 1)][gen_mask])

        selected_pred = torch.cat(selected_pred)
        selected_target = torch.cat(selected_target)

        if selected_pred.numel() == 0:
            loss = pred_bus.sum() * 0.0
        else:
            loss = F.mse_loss(selected_pred, selected_target, reduction=self.reduction)
        return {"loss": loss, "Masked MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedGenMSE")
class MaskedGenMSE(torch.nn.Module):
    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
    ):
        loss = F.mse_loss(
            pred_dict["gen"][mask_dict["gen"][:, : (PG_H + 1)]],
            target_dict["gen"][mask_dict["gen"][:, : (PG_H + 1)]],
            reduction=self.reduction,
        )
        return {"loss": loss, "Masked generator MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedBusMSE")
class MaskedBusMSE(torch.nn.Module):
    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"
        self.args = args

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
    ):
        if self.args.task == "OptimalPowerFlow":
            pred_cols = [VM_OUT, VA_OUT, QG_OUT]
            target_cols = [VM_H, VA_H, QG_H]
        else:
            pred_cols = [VM_OUT, VA_OUT]
            target_cols = [VM_H, VA_H]

        pred_bus = pred_dict["bus"][:, pred_cols]  # shape: [N, 3]
        target_bus = target_dict["bus"][:, target_cols]

        mask = mask_dict["bus"][:, target_cols]

        loss = F.mse_loss(
            pred_bus[mask],
            target_bus[mask],
            reduction=self.reduction,
        )
        return {"loss": loss, "Masked bus MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MSE")
class MSELoss(BaseLoss):
    """Standard Mean Squared Error loss."""

    def __init__(self, loss_args, args):
        super(MSELoss, self).__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return {"loss": loss, "MSE loss": loss.detach()}


class MixedLoss(BaseLoss):
    """
    Combines multiple loss functions with weighted sum.

    Args:
        loss_functions (list[nn.Module]): List of loss functions.
        weights (list[float]): Corresponding weights for each loss function.
    """

    def __init__(self, loss_functions, weights):
        super(MixedLoss, self).__init__()

        if len(loss_functions) != len(weights):
            raise ValueError(
                "The number of loss functions must match the number of weights.",
            )

        self.loss_functions = nn.ModuleList(loss_functions)
        self.weights = weights

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        """
        Compute the weighted sum of all specified losses.

        Parameters:

        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.

        Returns:
        - A dictionary with the total loss and individual losses.
        """
        total_loss = 0.0
        loss_details = {}

        for i, loss_fn in enumerate(self.loss_functions):
            loss_output = loss_fn(
                pred,
                target,
                edge_index,
                edge_attr,
                mask,
                model,
            )

            # Assume each loss function returns a dictionary with a "loss" key
            individual_loss = loss_output.pop("loss")
            weighted_loss = self.weights[i] * individual_loss

            total_loss += weighted_loss

            # Add other keys from the loss output to the details
            for key, val in loss_output.items():
                loss_details[key] = val

        loss_details["loss"] = total_loss
        return loss_details




@LOSS_REGISTRY.register("LayeredWeightedPhysics")
class LayeredWeightedPhysicsLoss(BaseLoss):
    def __init__(self, loss_args, args) -> None:
        super().__init__()
        self.base_weight = loss_args.base_weight

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        total_loss = 0.0
        loss_details = {}

        layer_keys = sorted(model.layer_residuals.keys())
        L = len(layer_keys)

        # Compute raw weights (geometric decay)
        raw_weights = [self.base_weight ** (L - idx - 1) for idx in range(L)]

        # Normalize so weights sum to 1
        weight_sum = sum(raw_weights)
        norm_weights = [w / weight_sum for w in raw_weights]

        for key, weight in zip(layer_keys, norm_weights):
            residual = model.layer_residuals[key]
            total_loss = total_loss + weight * residual
            loss_details[f"layer_{key}_residual"] = residual.item()
            loss_details[f"layer_{key}_weight"] = weight

        loss_details["loss"] = total_loss
        loss_details["Layered Weighted Physics Loss"] = total_loss.item()
        return loss_details


@LOSS_REGISTRY.register("LossPerDim")
class LossPerDim(BaseLoss):
    def __init__(self, loss_args, args):
        super(LossPerDim, self).__init__()
        self.reduction = "mean"
        self.loss_str = loss_args.loss_str
        self.dim = loss_args.dim
        if self.dim not in ["VM", "VA", "P_in", "Q_in"]:
            raise ValueError(
                f"LossPerDim initialized with not valid dim: {self.dim}",
            )

        elif self.loss_str not in ["MAE", "MSE"]:
            raise ValueError(
                f"LossPerDim initialized with not valid loss_str: {self.loss_str}",
            )

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
    ):
        if self.dim == "VM":
            temp_pred = pred_dict["bus"][:, VM_OUT]
            temp_target = target_dict["bus"][:, VM_H]
        elif self.dim == "VA":
            temp_pred = pred_dict["bus"][:, VA_OUT]
            temp_target = target_dict["bus"][:, VA_H]
        elif self.dim == "P_in":
            temp_pred = pred_dict["bus"][:, PG_OUT]
            num_bus = temp_pred.size(0)
            gen_to_bus_index = edge_index[("gen", "connected_to", "bus")]
            temp_gen = scatter_add(
                target_dict["gen"][:, PG_H],
                gen_to_bus_index[1, :],
                dim=0,
                dim_size=num_bus,
            )
            temp_target = temp_gen - target_dict["bus"][:, PD_H]
        elif self.dim == "Q_in":
            temp_pred = pred_dict["bus"][:, QG_OUT]
            temp_target = target_dict["bus"][:, QG_H] - target_dict["bus"][:, QD_H]

        mse_loss = F.mse_loss(temp_pred, temp_target, reduction=self.reduction)
        mae_loss = F.l1_loss(temp_pred, temp_target, reduction=self.reduction)

        loss = mse_loss if self.loss_str == "mse" else mae_loss
        return {
            "loss": loss,
            f"MSE loss {self.dim}": mse_loss.detach(),
            f"MAE loss {self.dim}": mae_loss.detach(),
        }

#######################
# ---------------------------------------------------------------------------
# VLD (Voltage Loss Detection) task loss
# ---------------------------------------------------------------------------
# NOTE (integration fix): the registry decorator was previously attached to the
# helper ``_node_pred_tensor`` below, which (a) registered the wrong object and
# (b) would have collided with the class decorator, raising KeyError at import
# because Registry.register forbids duplicate names. The decorator now lives on
# the working VLDTopologyLoss class further down.


@LOSS_REGISTRY.register("VLDTopologyLoss")
class VLDTopologyLoss(BaseLoss):
    """
    Topology-first voltage loss detection objective.

    Expected bus tensor layouts:
      pred_dict["bus"]   : [Vm, Va, Pg, Qg, status_logit]
      target_dict["bus"] : [Pd, Qd, Qg, Vm, Va, bus_status_target]
      x_dict["bus"]      : standard bus features + [bus_base_status, bus_contingency]
      edge_attr          : standard edge attrs + [branch_base_status, branch_contingency]
    """

    def __init__(self, loss_args, args):
        super().__init__()
        self.args = args
        self.input_state_threshold = getattr(loss_args, "input_state_threshold", 0.5)
        self.prediction_threshold = getattr(loss_args, "prediction_threshold", 0.5)
        self.topology_weight = getattr(loss_args, "topology_weight", 1.0)
        self.target_anchor_weight = getattr(loss_args, "target_anchor_weight", 0.25)
        self.off_vm_weight = getattr(loss_args, "off_vm_weight", 1.0)
        self.on_vm_weight = getattr(loss_args, "on_vm_weight", 0.5)
        self.unreachable_l1_weight = getattr(loss_args, "unreachable_l1_weight", 1.0)
        self.topology_confidence_gamma = getattr(loss_args, "topology_confidence_gamma", 10.0)

    @staticmethod
    def _build_graph_reachability(
        num_bus,
        edge_index,
        edge_attr,
        bus_x,
        device,
        threshold=0.5,
    ):
        """
        Build hard reachability labels from base-status and contingency indicators.

        A bus is considered initially available if base_status is on and it is not
        directly hit by contingency. A branch is traversable if base branch status
        is on and it is not hit by contingency.
        """
        bus_base = (bus_x[:, BUS_BASE_STATUS_H] > threshold)
        bus_hit = (bus_x[:, BUS_CONT_H] > threshold)
        bus_available = bus_base & (~bus_hit)

        src, dst = edge_index
        branch_base = (edge_attr[:, BRANCH_BASE_STATUS_E] > threshold)
        branch_hit = (edge_attr[:, BRANCH_CONT_E] > threshold)
        branch_available = branch_base & (~branch_hit)

        reachable = torch.zeros(num_bus, dtype=torch.bool, device=device)

        # Seeds: all buses that remain available after direct contingency.
        seed_nodes = torch.where(bus_available)[0]
        if seed_nodes.numel() == 0:
            return reachable.float(), bus_available.float()

        reachable[seed_nodes] = True

        changed = True
        while changed:
            prev = reachable.clone()
            active_edges = branch_available & reachable[src]
            reachable[dst[active_edges]] = True
            changed = not torch.equal(prev, reachable)

        return reachable.float(), bus_available.float()

    def forward(
            self,
            pred_dict,
            target_dict,
            edge_index_dict,
            edge_attr_dict,
            mask_dict,
            model=None,
    ):
        bus_pred = pred_dict["bus"]
        bus_target = target_dict["bus"]
        bus_x = model.latest_x_dict["bus"] if hasattr(model, "latest_x_dict") else None

        if bus_x is None:
            raise RuntimeError(
                "VLDTopologyLoss requires model.latest_x_dict['bus']. "
                "Store x_dict on the model inside the forward pass."
            )

        if bus_pred.size(1) <= BUS_STATUS_LOGIT_OUT:
            raise ValueError(
                "VLDTopologyLoss expects bus predictions to include a status logit "
                f"at column {BUS_STATUS_LOGIT_OUT}."
            )

        edge_index = edge_index_dict[("bus", "connects", "bus")]
        edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        num_bus = bus_pred.size(0)
        device = bus_pred.device

        topo_target, bus_available = self._build_graph_reachability(
            num_bus=num_bus,
            edge_index=edge_index,
            edge_attr=edge_attr,
            bus_x=bus_x,
            device=device,
            threshold=self.input_state_threshold,
        )

        status_logit = bus_pred[:, BUS_STATUS_LOGIT_OUT]
        status_prob = torch.sigmoid(status_logit)

        target_status = bus_target[:, BUS_STATUS_TARGET].float()
        target_vm = bus_target[:, VM_H].float()
        pred_vm = bus_pred[:, VM_OUT].float()

        topology_confidence = torch.exp(
            -self.topology_confidence_gamma * torch.abs(bus_available - topo_target)
        )

        topology_bce_raw = F.binary_cross_entropy_with_logits(
            status_logit,
            topo_target,
            reduction="none",
        )
        topology_bce = (topology_confidence * topology_bce_raw).mean()

        target_anchor_bce = F.binary_cross_entropy_with_logits(
            status_logit,
            target_status,
            reduction="mean",
        )

        unreachable_mask = (topo_target < 0.5).float()
        reachable_mask = (topo_target >= 0.5).float()

        off_vm_l2 = ((pred_vm ** 2) * unreachable_mask).sum() / unreachable_mask.sum().clamp_min(1.0)
        off_vm_l1 = (pred_vm.abs() * unreachable_mask).sum() / unreachable_mask.sum().clamp_min(1.0)
        off_vm_loss = off_vm_l2 + self.unreachable_l1_weight * off_vm_l1

        on_vm_sq = ((pred_vm - target_vm) ** 2) * reachable_mask * status_prob.detach()
        on_vm_loss = on_vm_sq.sum() / (reachable_mask * status_prob.detach()).sum().clamp_min(1.0)

        total_loss = (
            self.topology_weight * topology_bce
            + self.target_anchor_weight * target_anchor_bce
            + self.off_vm_weight * off_vm_loss
            + self.on_vm_weight * on_vm_loss
        )

        pred_status = (status_prob >= self.prediction_threshold).float()
        topo_acc = (pred_status == topo_target).float().mean()
        target_acc = (pred_status == target_status).float().mean()

        return {
            "loss": total_loss,
            "VLD Topology BCE": topology_bce.detach(),
            "VLD Target Anchor BCE": target_anchor_bce.detach(),
            "VLD Off Vm Loss": off_vm_loss.detach(),
            "VLD On Vm Loss": on_vm_loss.detach(),
            "VLD Topology Accuracy": topo_acc.detach(),
            "VLD Target Accuracy": target_acc.detach(),
        }
#######################

# ---------------------------------------------------------------------------
# Physics-based losses carried over unchanged from gridfm-graphkit.
# ---------------------------------------------------------------------------
@LOSS_REGISTRY.register("PBE")
class PBELoss(BaseLoss):
    """Loss based on the Power Balance Equations (heterogeneous convention)."""

    def __init__(self, loss_args=None, args=None, visualization=False):
        super(PBELoss, self).__init__()
        self.visualization = (
            getattr(loss_args, "visualization", False)
            if loss_args is not None
            else visualization
        )

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        model=None,
        x_dict=None,
    ):
        pred_bus = pred_dict["bus"]
        target_bus = target_dict["bus"]
        num_bus = target_bus.size(0)

        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]
        mask_bus = mask_dict["bus"]

        Vm_pred = pred_bus[:, VM_OUT]
        Va_pred = pred_bus[:, VA_OUT]
        Vm_target = target_bus[:, VM_H]
        Va_target = target_bus[:, VA_H]

        mask_Vm = mask_bus[:, VM_H]
        mask_Va = mask_bus[:, VA_H]

        V_m = torch.where(mask_Vm, Vm_pred, Vm_target)
        V_a = torch.where(mask_Va, Va_pred, Va_target)

        V = V_m * torch.exp(1j * V_a)
        V_conj = torch.conj(V)

        edge_offdiag = bus_edge_attr[:, YFT_TF_R] + 1j * bus_edge_attr[:, YFT_TF_I]

        Y_diag_r = scatter_add(
            bus_edge_attr[:, YFF_TT_R], bus_edge_index[0], dim=0, dim_size=num_bus,
        )
        Y_diag_i = scatter_add(
            bus_edge_attr[:, YFF_TT_I], bus_edge_index[0], dim=0, dim_size=num_bus,
        )
        Y_diag = Y_diag_r + 1j * Y_diag_i

        if x_dict is not None:
            bus_orig = x_dict["bus"]
            Y_diag = Y_diag + bus_orig[:, GS] + 1j * bus_orig[:, BS]

        diag_idx = torch.arange(num_bus, device=bus_edge_index.device)
        full_edge_index = torch.cat(
            [bus_edge_index, torch.stack([diag_idx, diag_idx])], dim=1,
        )
        full_edge_values = torch.cat([edge_offdiag, Y_diag])

        Y_bus_sparse = to_torch_coo_tensor(
            full_edge_index, full_edge_values, size=(num_bus, num_bus),
        )
        Y_bus_conj = torch.conj(Y_bus_sparse)

        S_injection = V * (Y_bus_conj @ V_conj)

        gen_to_bus_ei = edge_index_dict[("gen", "connected_to", "bus")]
        target_pg_agg = scatter_add(
            target_dict["gen"][:, PG_H], gen_to_bus_ei[1], dim=0, dim_size=num_bus,
        )
        gen_pg_masked = mask_dict["gen"][:, PG_H].float()
        any_gen_masked = (
            scatter_add(gen_pg_masked, gen_to_bus_ei[1], dim=0, dim_size=num_bus) > 0
        )
        Pg_per_bus = torch.where(any_gen_masked, pred_bus[:, PG_OUT], target_pg_agg)

        # Guard on BUS_STATUS_TARGET (index 5) rather than PD_OUT/QD_OUT so a VLD
        # bus output ([VM, VA, PG, QG, status_logit], 5 columns) never treats the
        # status logit at column 4 as a predicted load-power value. Only base-task
        # outputs (>= 6 columns) carry genuine PD/QD predictions.
        if pred_bus.size(1) > BUS_STATUS_TARGET:
            Pd = torch.where(mask_bus[:, PD_H], pred_bus[:, PD_OUT], target_bus[:, PD_H])
            Qd = torch.where(mask_bus[:, QD_H], pred_bus[:, QD_OUT], target_bus[:, QD_H])
        else:
            Pd = target_bus[:, PD_H]
            Qd = target_bus[:, QD_H]
        if pred_bus.size(1) > QG_OUT:
            Qg = torch.where(mask_bus[:, QG_H], pred_bus[:, QG_OUT], target_bus[:, QG_H])
        else:
            Qg = target_bus[:, QG_H]

        net_P = Pg_per_bus - Pd
        net_Q = Qg - Qd
        S_net = net_P + 1j * net_Q

        loss = torch.mean(torch.abs(S_net - S_injection))
        real_loss = torch.mean(torch.abs(torch.real(S_net - S_injection)))
        imag_loss = torch.mean(torch.abs(torch.imag(S_net - S_injection)))

        result = {
            "loss": loss,
            "Power loss in p.u.": loss.detach(),
            "Active Power Loss in p.u.": real_loss.detach(),
            "Reactive Power Loss in p.u.": imag_loss.detach(),
        }
        if self.visualization:
            result["Nodal Active Power Loss in p.u."] = torch.abs(
                torch.real(S_net - S_injection),
            )
            result["Nodal Reactive Power Loss in p.u."] = torch.abs(
                torch.imag(S_net - S_injection),
            )
        return result


@LOSS_REGISTRY.register("QgViolationPenalty")
class QgViolationPenaltyLoss(BaseLoss):
    """Penalty on generator reactive-power limit violations."""

    def __init__(self, loss_args=None, args=None):
        super().__init__()

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
    ):
        Qg_pred = pred["bus"][:, QG_OUT]
        Qg_max = x_dict["bus"][:, MAX_QG_H]
        Qg_min = x_dict["bus"][:, MIN_QG_H]

        max_penalty_mask = Qg_pred > Qg_max
        min_penalty_mask = Qg_pred < Qg_min

        loss = 0.0
        Qg_over = F.relu(Qg_pred - Qg_max)
        Qg_under = F.relu(Qg_min - Qg_pred)

        Qg_over = Qg_over[max_penalty_mask].mean()
        Qg_under = Qg_under[min_penalty_mask].mean()

        if Qg_over != Qg_over:
            Qg_over = 0.0
        if Qg_under != Qg_under:
            Qg_under = 0.0

        penalty_loss = Qg_over + Qg_under
        loss += penalty_loss

        try:
            output = {"loss": loss, "Qg Violation Penalty loss": loss.detach()}
        except Exception:
            output = {"loss": loss, "Qg Violation Penalty loss": loss}

        return output
