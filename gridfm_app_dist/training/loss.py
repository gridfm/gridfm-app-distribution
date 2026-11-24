import torch.nn.functional as F
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from gridfm_app_dist.io.registries import LOSS_REGISTRY
from torch_scatter import scatter_add
from torch_geometric.utils import to_torch_coo_tensor

from gridfm_app_dist.datasets.globals import (
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
    # Indices of features of the VLD task
    # Bus feature indices
    REF_H,
    PV_H,
    BUS_BASE_STATUS_H,
    BUS_CONT_H,
    # B_ON,
    # Branch feature indices
    BRANCH_BASE_STATUS_E,
    BRANCH_CONT_E,
    # PBE slack power from the label
    PG_BUS_H,
)


def _task_name(args):
    """``args.task.task_name`` or None (args may be None, e.g. in diagnostics)."""
    return getattr(getattr(args, "task", None), "task_name", None)


def _is_vld(args):
    return _task_name(args) == "VoltageLossDetection"


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

    Channels (``loss_args.channels``)
    ---------------------------------
    Which masked entries enter the MSE: any of ``Vm``, ``Va``, ``Qg`` (bus) and
    ``gen_Pg`` (generator rows). Default: all four, except for
    VoltageLossDetection, where the default is ``[Vm, Va]``.

    For VLD the bus ``Qg`` at PV/REF buses is not a free prediction: the physics
    decoder computes it from the predicted voltages (``Q_in + Qd - q_shunt``),
    and the masked generator ``Pg`` rows are the slack generators, whose value is
    of the order of the whole network load in p.u. Both are therefore several
    orders of magnitude larger than Vm/Va and dominate the MSE and its gradient
    on large or unfamiliar networks (MaskedMSE up to 3e7 and Vm-decoder gradient
    norms up to 2e9 were measured on kerber_2). The voltage channels carry the reconstruction signal;
    the power balance is enforced by PBE.
    """

    # (prediction column, label/mask column) for the masked reconstruction channels.
    BUS_COLS = ((VM_OUT, VM_H), (VA_OUT, VA_H), (QG_OUT, QG_H))
    BUS_CHANNELS = {"Vm": (VM_OUT, VM_H), "Va": (VA_OUT, VA_H), "Qg": (QG_OUT, QG_H)}
    ALL_CHANNELS = ("Vm", "Va", "Qg", "gen_Pg")
    VLD_DEFAULT_CHANNELS = ("Vm", "Va")

    def __init__(self, loss_args, args):
        super(MaskedMSELoss, self).__init__()
        self.reduction = "mean"
        channels = (
            getattr(loss_args, "channels", None) if loss_args is not None else None
        )
        if channels is None:
            channels = self.VLD_DEFAULT_CHANNELS if _is_vld(args) else self.ALL_CHANNELS
        if isinstance(channels, str):
            channels = [channels]
        unknown = [c for c in channels if c not in self.ALL_CHANNELS]
        if unknown or not channels:
            raise ValueError(
                f"MaskedMSE loss_args.channels must be a non-empty subset of "
                f"{list(self.ALL_CHANNELS)}, got {channels!r}.",
            )
        self.channels = tuple(c for c in self.ALL_CHANNELS if c in channels)
        self.bus_cols = tuple(
            self.BUS_CHANNELS[c] for c in self.channels if c in self.BUS_CHANNELS
        )
        self.use_gen_pg = "gen_Pg" in self.channels

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
        selected_pred = [pred_bus[:, p][mask_bus[:, t]] for p, t in self.bus_cols]
        selected_target = [target_bus[:, t][mask_bus[:, t]] for _, t in self.bus_cols]

        if self.use_gen_pg:
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


# ---------------------------------------------------------------------------
# Voltage Loss Detection (VLD) task loss - Topology-first voltage loss detection
# ---------------------------------------------------------------------------


@LOSS_REGISTRY.register("VLDTopologyLoss")
class VLDTopologyLoss(BaseLoss):
    """
    Expected bus tensor layouts:
      pred_dict["bus"]   : [Vm, Va, Pg, Qg, status_logit]
      target_dict["bus"] : [Pd, Qd, Qg, Vm, Va, bus_status_target]
      x_dict["bus"]      : standard bus features + [bus_base_status, bus_contingency]
      edge_attr          : standard edge attrs + [branch_base_status, branch_contingency]

    y_reach is the hard REF-reachability built from the input flags (BFS over
    available branches through available buses, ``_build_graph_reachability``).
    It is computed and used inside this loss only;

    Status head (status logit z), two BCE terms:
      * topology term (primary) - BCE(z, y_reach) on EVERY bus, class-weighted on
        y_reach: w_off = n_on / n_off of y_reach (eq. 6, EMA-smoothed) for
        unreachable buses, 1 otherwise.
      * anchor term (secondary) - BCE(z, y_target) against the dataset label, by
        default only on conflict buses C = {i : y_reach != y_target}
        (``anchor_scope: conflict``), unweighted by class and normalised by the
        topology term's denominator. With target_anchor_weight < 1 the decision
        on a conflict bus still follows reachability at the 1 / (1 + w_off)
        threshold (the label only softens the probability). With correctly
        generated data C is empty and the anchor is 0. ``anchor_scope: all`` applies the anchor to every bus,
        class-weighted on the label

    Voltage head (Vm), independent of the status head:
      * off-Vm term - Vm -> 0 (L2 + lambda_1 * L1) on buses with y_reach = 0;
      * on-Vm term  - (Vm - Vm_target)^2 on buses with y_reach = 1.
      Both masks come from y_reach only; neither the status logit nor its
      probability enters the voltage terms

    Decision threshold: weighting the 0-class by w moves the BCE optimum from the
    posterior q to q / (q + w (1 - q)), so the threshold is 1 / (1 + w_off), which
    restores the boundary q = 0.5 and tracks w_off. Use the same rule at inference
    (returned as "VLD Decision Threshold"; w_off is stored in ``ema_off_weight``).

    Tunable parameters:
      topology_weight, target_anchor_weight, off_vm_weight, on_vm_weight,
      unreachable_l1_weight  - the lambda_* of eq. (18) and lambda_1 of eq. (15)
      anchor_scope           - "conflict" (default) or "all"
      topology_confidence_gamma - gamma of eq. (8), 0 disables
      reachability_sources   - "ref" or "ref_and_gen"
      unreachable_pos_weight - "auto" (n_on/n_off, EMA-smoothed) or a number
      max_pos_weight, pos_weight_momentum, input_state_threshold
    """

    def __init__(self, loss_args, args):
        super().__init__()
        self.args = args
        self.input_state_threshold = getattr(loss_args, "input_state_threshold", 0.5)
        self.topology_weight = getattr(loss_args, "topology_weight", 1.0)
        self.target_anchor_weight = getattr(loss_args, "target_anchor_weight", 0.25)
        self.off_vm_weight = getattr(loss_args, "off_vm_weight", 1.0)
        self.on_vm_weight = getattr(loss_args, "on_vm_weight", 0.5)
        self.unreachable_l1_weight = getattr(loss_args, "unreachable_l1_weight", 1.0)
        self.topology_confidence_gamma = getattr(
            loss_args,
            "topology_confidence_gamma",
            0.0,
        )

        # Where the dataset-label anchor applies: "conflict" (secondary anchor on
        # buses where the label disagrees with reachability) or "all" buses.
        self.anchor_scope = getattr(loss_args, "anchor_scope", "conflict")
        if self.anchor_scope not in ("conflict", "all"):
            raise ValueError(
                f"loss.anchor_scope must be 'conflict' or 'all', got {self.anchor_scope!r}.",
            )

        # Which buses count as energisation sources for the reachability BFS.
        #   "ref"         only the slack / ext_grid buses (REF_H).
        #   "ref_and_gen" REF_H or PV_H, i.e. islands with local generation are treated as energised.
        #                 Use only for grid-forming DG or  intentional-islanding studies.
        self.reachability_sources = getattr(loss_args, "reachability_sources", "ref")
        if self.reachability_sources not in ("ref", "ref_and_gen"):
            raise ValueError(
                f"loss.reachability_sources must be 'ref' or 'ref_and_gen', got {self.reachability_sources!r}.",
            )

        # Up-weight the de-energised class in the topology BCE.
        # With correct labels the de-energised buses are a minority (severely so on large meshed networks),
        # and an unweighted BCE is minimised by predicting "energised" everywhere.
        # "auto" balances per batch; 1.0 disables.
        self.unreachable_pos_weight = getattr(
            loss_args,
            "unreachable_pos_weight",
            "auto",
        )
        if self.unreachable_pos_weight != "auto":
            try:
                self.unreachable_pos_weight = float(self.unreachable_pos_weight)
            except (TypeError, ValueError):
                raise ValueError(
                    f"floss.unreachable_pos_weight must be 'auto' or a number, got {self.unreachable_pos_weight!r}.",
                ) from None
        self.max_pos_weight = float(getattr(loss_args, "max_pos_weight", 50.0))

        # EMA momentum for the "auto" class weight. Because the decision threshold is 1/(1+w_off), w_off must be stable across batches;
        # the EMA tracks the dataset-level ratio so a batch with a single de-energised bus can neither dominate the gradient nor move the threshold.
        # 0.0 falls back to the per-batch estimate. Updated only in training mode; frozen (and checkpointed) for evaluation.
        self.pos_weight_momentum = float(
            getattr(loss_args, "pos_weight_momentum", 0.99),
        )
        if not 0.0 <= self.pos_weight_momentum < 1.0:
            raise ValueError(
                f"loss.pos_weight_momentum must be in [0, 1), got {self.pos_weight_momentum!r}.",
            )
        self.register_buffer("ema_off_weight", torch.ones(()))
        self.register_buffer("ema_initialised", torch.zeros((), dtype=torch.bool))

    def _resolve_off_weight(self, sup_status, device):
        """Minority-class weight w_off EMA-smoothed."""
        n_off = (sup_status < 0.5).sum()
        n_on = (sup_status >= 0.5).sum()
        if self.unreachable_pos_weight != "auto":
            return torch.as_tensor(
                float(self.unreachable_pos_weight),
                device=device,
            ).clamp_min(1.0)

        if n_off > 0:
            batch_weight = (n_on.float() / n_off.float()).clamp(
                1.0,
                self.max_pos_weight,
            )
        else:
            batch_weight = None

        if self.pos_weight_momentum == 0.0:
            return (
                batch_weight
                if batch_weight is not None
                else torch.ones((), device=device)
            )

        if self.training and batch_weight is not None:
            with torch.no_grad():
                if bool(self.ema_initialised):
                    self.ema_off_weight.mul_(self.pos_weight_momentum).add_(
                        (1.0 - self.pos_weight_momentum)
                        * batch_weight.to(self.ema_off_weight),
                    )
                else:
                    self.ema_off_weight.copy_(batch_weight)
                    self.ema_initialised.fill_(True)
        if bool(self.ema_initialised):
            return self.ema_off_weight.to(device)
        return (
            batch_weight if batch_weight is not None else torch.ones((), device=device)
        )

    def reset_class_weight_ema(self):
        """
        Forget the EMA of the class weight w_off (and so the decision threshold).

        Called when fine-tuning on a new network: the checkpoint carries the
        previous network's w_off, which would otherwise set the class weight and
        the 1 / (1 + w_off) threshold of the new stage until the EMA catches up.
        After a reset the first training batch initialises the EMA from the new
        network; until then the per-batch value is used.
        Returns the w_off value that was discarded (None if it was not set).
        """
        previous = float(self.ema_off_weight) if bool(self.ema_initialised) else None
        with torch.no_grad():
            self.ema_off_weight.fill_(1.0)
            self.ema_initialised.fill_(False)
        return previous

    def decision_threshold(self):
        """
        Decision threshold theta = 1 / (1 + w_off) for the status probability,
        or ``None`` when no training-time w_off is available.

        Weighting the 0-class by w moves the BCE optimum from the posterior q to q / (q + w (1 - q)); thresholding at 1 / (1 + w) maps it back to q = 0.5.
        This is the threshold the loss uses for its own metrics and the one the task must apply at test / predict time.
        It is derived from the fixed ``unreachable_pos_weight`` when one is configured, otherwise from the EMA buffer ``ema_off_weight`` (saved in the checkpoint).
        ``None`` means the EMA was never initialised (an old checkpoint, or ``pos_weight_momentum == 0``);
        callers should then fall back to the per-batch value returned as "VLD Decision Threshold".
        """
        if self.unreachable_pos_weight != "auto":
            w = max(float(self.unreachable_pos_weight), 1.0)
            return 1.0 / (1.0 + w)
        if self.pos_weight_momentum > 0.0 and bool(self.ema_initialised):
            return 1.0 / (1.0 + float(self.ema_off_weight))
        return None

    @staticmethod
    def _build_graph_reachability(
        num_bus,
        edge_index,
        edge_attr,
        bus_x,
        device,
        threshold=0.5,
        sources="ref",
    ):
        """
        Build hard reachability labels from base-status and contingency indicators.

        A bus is energised if:
        (a) it is itself available - base status on and not directly hit by a contingency; AND
        (b) there is a path of available branches through available buses from an energisation SOURCE to it.
        """
        bus_base = bus_x[:, BUS_BASE_STATUS_H] > threshold
        bus_hit = bus_x[:, BUS_CONT_H] > threshold
        bus_available = bus_base & (~bus_hit)

        src, dst = edge_index
        branch_base = edge_attr[:, BRANCH_BASE_STATUS_E] > threshold
        branch_hit = edge_attr[:, BRANCH_CONT_E] > threshold
        branch_available = branch_base & (~branch_hit)

        reachable = torch.zeros(num_bus, dtype=torch.bool, device=device)
        is_source = bus_x[:, REF_H] > threshold
        if sources == "ref_and_gen":
            is_source = is_source | (bus_x[:, PV_H] > threshold)
        seeds = bus_available & is_source
        if not bool(seeds.any()):
            # Whole-graph blackout: nothing is energised. Returning all-zero is
            # correct, not a degenerate fallback.
            return reachable.float(), bus_available.float()
        reachable = reachable | seeds

        # Symmetrise: do not depend on both directions being present in the data.
        e_src = torch.cat([src, dst])
        e_dst = torch.cat([dst, src])
        e_ok = torch.cat([branch_available, branch_available])

        for _ in range(num_bus):
            active_edges = e_ok & reachable[e_src]
            frontier = torch.zeros_like(reachable)
            frontier[e_dst[active_edges]] = True
            # A bus that is itself out of service cannot become energised, and
            # cannot pass energy on.
            frontier = frontier & bus_available
            updated = reachable | frontier
            if torch.equal(updated, reachable):
                break
            reachable = updated
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
                "VLDTopologyLoss requires model.latest_x_dict['bus']. Store x_dict on the model inside the forward pass.",
            )

        if bus_pred.size(1) <= BUS_STATUS_LOGIT_OUT:
            raise ValueError(
                f"VLDTopologyLoss expects bus predictions to include a status logit at column {BUS_STATUS_LOGIT_OUT}.",
            )

        edge_index = edge_index_dict[("bus", "connects", "bus")]
        edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        device = bus_pred.device

        # REF-reachability from the input flags (used by this loss only).
        topo_target, bus_available = self._build_graph_reachability(
            num_bus=bus_pred.size(0),
            edge_index=edge_index,
            edge_attr=edge_attr,
            bus_x=bus_x,
            device=device,
            threshold=self.input_state_threshold,
            sources=self.reachability_sources,
        )

        status_logit = bus_pred[:, BUS_STATUS_LOGIT_OUT]
        status_prob = torch.sigmoid(status_logit)

        target_status = bus_target[:, BUS_STATUS_TARGET].float()
        target_vm = bus_target[:, VM_H].float()
        pred_vm = bus_pred[:, VM_OUT].float()

        # Conflict buses: BFS reachability and the dataset label disagree.
        conflict = (topo_target < 0.5) != (target_status < 0.5)

        # Per-sample weights on the topology BCE.
        if self.topology_confidence_gamma:
            topology_confidence = torch.exp(
                -self.topology_confidence_gamma
                * torch.abs(bus_available - topo_target),
            )
        else:
            topology_confidence = torch.ones_like(topo_target)

        # Class balance. De-energised buses are the minority class and the one that matters;
        # an unweighted BCE over correct labels is very nearly minimised by predicting "energised" everywhere.
        off_weight = self._resolve_off_weight(topo_target, device)
        ones = torch.ones_like(topo_target)
        class_weight = torch.where(
            topo_target < 0.5,
            off_weight.expand_as(topo_target),
            ones,
        )

        # omega_i: the topology (reachability) term is the primary status
        # supervision on EVERY bus, conflict buses included.
        topology_weights = topology_confidence * class_weight
        topology_bce_raw = F.binary_cross_entropy_with_logits(
            status_logit,
            topo_target,
            reduction="none",
        )
        topology_norm = topology_weights.sum().clamp_min(1e-8)
        topology_bce = (topology_weights * topology_bce_raw).sum() / topology_norm

        # nu_i,: dataset-label anchor, secondary.
        anchor_bce_raw = F.binary_cross_entropy_with_logits(
            status_logit,
            target_status,
            reduction="none",
        )
        if self.anchor_scope == "conflict":
            # Only where the label disagrees with reachability, WITHOUT the class
            # weight and over the topology term's denominator. On a conflict bus the
            # optimum then lies on the reachability side of the 1 / (1 + w_off)
            # decision threshold whenever target_anchor_weight < 1, in both
            # directions (reach 0 / label 1 and reach 1 / label 0).
            anchor_weights = topology_confidence * conflict.float()
            target_anchor_bce = (anchor_weights * anchor_bce_raw).sum() / topology_norm
        else:
            anchor_weights = torch.where(
                target_status < 0.5,
                off_weight.expand_as(target_status),
                ones,
            )
            target_anchor_bce = (anchor_weights * anchor_bce_raw).sum() / (
                anchor_weights.sum().clamp_min(1e-8)
            )

        # Voltage supervision is keyed on reachability ONLY.
        unreachable_mask = (topo_target < 0.5).float()
        reachable_mask = (topo_target >= 0.5).float()

        off_vm_l2 = (
            (pred_vm**2) * unreachable_mask
        ).sum() / unreachable_mask.sum().clamp_min(1.0)
        off_vm_l1 = (
            pred_vm.abs() * unreachable_mask
        ).sum() / unreachable_mask.sum().clamp_min(1.0)
        off_vm_loss = off_vm_l2 + self.unreachable_l1_weight * off_vm_l1

        on_vm_sq = ((pred_vm - target_vm) ** 2) * reachable_mask
        on_vm_loss = on_vm_sq.sum() / reachable_mask.sum().clamp_min(1.0)

        total_loss = (
            self.topology_weight * topology_bce
            + self.target_anchor_weight * target_anchor_bce
            + self.off_vm_weight * off_vm_loss
            + self.on_vm_weight * on_vm_loss
        )

        # Balanced decision threshold. The class weight moves the BCE optimum from q to q / (q + w (1 - q));
        # thresholding at 1 / (1 + w) maps that back to the boundary q = 0.5, and follows w_off as it moves.
        threshold = 1.0 / (1.0 + off_weight.detach().float())
        pred_status = (status_prob >= threshold).float()
        topo_acc = (pred_status == topo_target).float().mean()
        target_acc = (pred_status == target_status).float().mean()

        # Accuracy on this task is dominated by the energised majority class and- reads ~99% for a model that never predicts an outage.
        # Report recall and precision on the DE-ENERGISED class
        with torch.no_grad():
            off_true = target_status < 0.5
            on_true = ~off_true
            off_pred = pred_status < 0.5
            tp = (off_true & off_pred).sum().float()
            recall_off = tp / off_true.sum().clamp_min(1).float()
            precision_off = tp / off_pred.sum().clamp_min(1).float()
            f1_off = (2 * precision_off * recall_off) / (
                precision_off + recall_off
            ).clamp_min(1e-8)

            # Energised recall = 1 - false-de-energisation rate.
            recall_on = (on_true & ~off_pred).sum().float() / on_true.sum().clamp_min(
                1,
            ).float()
            balanced_acc = 0.5 * (recall_off + recall_on)

            # Calibration on the buses that are unambiguously energised: BFS says reachable AND the label says 1.
            clear_on = on_true & (topo_target >= 0.5)
            mean_prob_clear_on = (
                status_prob * clear_on
            ).sum() / clear_on.sum().clamp_min(1)

            # Data-integrity check: with a correct BFS, the derived reachability label should agree with the dataset's own bus_status_target.
            # A persistent disagreement means one of the two is wrong - loss reports the data inconsistencies in the training curves
            # Can be skipped if data is 100% checked
            topo_vs_target = conflict.float().mean()

            # Split the disagreement by direction: reach=0/label=1 is the case that the legacy "zero" rule turned into false de-energisations.
            conflict_label_on = (conflict & on_true).float().mean()
            conflict_label_off = (conflict & off_true).float().mean()
            off_rate_reach = (topo_target < 0.5).float().mean()
            off_rate_label = off_true.float().mean()

        return {
            "loss": total_loss,
            "VLD Loss": total_loss.detach(),  # the VLD objective alone (LR-schedule monitor)
            "VLD Topology BCE": topology_bce.detach(),
            "VLD Target Anchor BCE": target_anchor_bce.detach(),
            "VLD Off Vm Loss": off_vm_loss.detach(),
            "VLD On Vm Loss": on_vm_loss.detach(),
            "VLD Topology Accuracy": topo_acc.detach(),
            "VLD Target Accuracy": target_acc.detach(),
            "VLD Balanced Accuracy": balanced_acc,
            "VLD Off Recall": recall_off,
            "VLD Off Precision": precision_off,
            "VLD Off F1": f1_off,
            "VLD On Recall": recall_on,
            "VLD Mean Prob (reachable & label 1)": mean_prob_clear_on,
            "VLD Off Rate (labels)": off_rate_label,
            "VLD Off Rate (reachability)": off_rate_reach,
            "VLD Topo-vs-Target Disagreement": topo_vs_target,
            "VLD Conflict Rate (reach 0, label 1)": conflict_label_on,
            "VLD Conflict Rate (reach 1, label 0)": conflict_label_off,
            "VLD Off Class Weight": off_weight.detach().float(),
            "VLD Decision Threshold": threshold.detach().float(),
        }

# ---------------------------------------------------------------------------
# Loss based in node residuals from Power Balance equations (PBE Loss)
# ---------------------------------------------------------------------------

@LOSS_REGISTRY.register("PBE")
class PBELoss(BaseLoss):
    """
    Options (``loss_args``). The defaults reproduce the original
    loss exactly for every task except VoltageLossDetection.

    ``slack_power``: ``label`` | ``gen_rows``
        Active power of buses whose generator Pg is not predicted.
        ``gen_rows`` (legacy) sums the generator rows only, so the ext_grid
        injection is missing on networks that model the slack as an ext_grid
        (about half of the PBE at a perfect prediction). ``label``
        takes the per-bus net generation label ``PG_BUS_H`` (generators +
        ext_grid) and, at REF buses, the labelled slack P and Q, so the slack
        bus contributes a real balance check: neither the irreducible error of
        ``gen_rows`` nor the residual that the decoder's slack Pg/Qg would make
        zero by construction. Default: ``label`` for VLD, ``gen_rows`` otherwise.
    ``merge_switches``: bool
        Closed bus-bus switches appear as in-service edges whose admittance
        entries are all zero; the two buses are one electrical node, so their
        separate balances are meaningless (only the sum balances). With
        ``true`` the residual is summed over each switch-connected group of
        buses (connected components over switch edges) before the magnitude is
        taken. Default: ``true`` for VLD, ``false`` otherwise.
    ``switch_tol``: float (default 0.0)
        An edge is a switch when ``|Yff_r|, |Yff_i|, |Yft_r|, |Yft_i| <= switch_tol``.
    ``bus_scope``: ``all`` | ``energised``
        ``energised`` averages over energised nodes only (label
        ``BUS_STATUS_TARGET``). De-energised buses keep their in-service load
        in the labels, so with ``all`` they add an irreducible floor equal to
        their unsupplied load, and away from V = 0 a pull unrelated to the VLD
        off-Vm term; they add no information. Default: ``energised`` for VLD,
        ``all`` otherwise.
    ``normalisation``: ``none`` | ``load``
        ``load`` divides each node's residual by the mean load magnitude
        ``|Pd + jQd|`` over the loaded buses of its own graph (scenario), so the
        loss is dimensionless and comparable across networks whose loads differ
        by orders of magnitude. Needs the per-bus graph index, which the task
        stores as ``model.latest_bus_batch``; without it the whole batch is
        treated as one graph. Default: ``none``.
    ``include_shunts``: bool (default true)
        Add the bus shunts ``GS + jBS`` to the diagonal of Y. Read from
        ``x_dict`` or ``model.latest_x_dict``. Only used by the new path.

    The reported ``Power loss in p.u.`` is always the un-normalised mean
    residual magnitude over the nodes in scope; the training value is ``loss``.
    """

    SLACK_MODES = ("label", "gen_rows")
    SCOPES = ("all", "energised")
    NORMALISATIONS = ("none", "load")

    def __init__(self, loss_args=None, args=None, visualization=False):
        super(PBELoss, self).__init__()
        self.visualization = (
            getattr(loss_args, "visualization", False)
            if loss_args is not None
            else visualization
        )
        vld = _is_vld(args)

        def opt(name, default):
            return (
                getattr(loss_args, name, default) if loss_args is not None else default
            )

        self.slack_power = opt("slack_power", "label" if vld else "gen_rows")
        self.merge_switches = bool(opt("merge_switches", vld))
        self.switch_tol = float(opt("switch_tol", 0.0))
        self.bus_scope = opt("bus_scope", "energised" if vld else "all")
        self.normalisation = opt("normalisation", "none")
        self.include_shunts = bool(opt("include_shunts", True))
        for name, value, allowed in (
            ("slack_power", self.slack_power, self.SLACK_MODES),
            ("bus_scope", self.bus_scope, self.SCOPES),
            ("normalisation", self.normalisation, self.NORMALISATIONS),
        ):
            if value not in allowed:
                raise ValueError(
                    f"PBE loss_args.{name} must be one of {list(allowed)}, got {value!r}.",
                )
        self.legacy = (
            self.slack_power == "gen_rows"
            and not self.merge_switches
            and self.bus_scope == "all"
            and self.normalisation == "none"
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
        if not self.legacy:
            return self._forward_nodes(
                pred_dict,
                target_dict,
                edge_index_dict,
                edge_attr_dict,
                mask_dict,
                model,
                x_dict,
            )
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
            bus_edge_attr[:, YFF_TT_R],
            bus_edge_index[0],
            dim=0,
            dim_size=num_bus,
        )
        Y_diag_i = scatter_add(
            bus_edge_attr[:, YFF_TT_I],
            bus_edge_index[0],
            dim=0,
            dim_size=num_bus,
        )
        Y_diag = Y_diag_r + 1j * Y_diag_i

        if x_dict is not None:
            bus_orig = x_dict["bus"]
            Y_diag = Y_diag + bus_orig[:, GS] + 1j * bus_orig[:, BS]

        diag_idx = torch.arange(num_bus, device=bus_edge_index.device)
        full_edge_index = torch.cat(
            [bus_edge_index, torch.stack([diag_idx, diag_idx])],
            dim=1,
        )
        full_edge_values = torch.cat([edge_offdiag, Y_diag])

        Y_bus_sparse = to_torch_coo_tensor(
            full_edge_index,
            full_edge_values,
            size=(num_bus, num_bus),
        )
        Y_bus_conj = torch.conj(Y_bus_sparse)

        S_injection = V * (Y_bus_conj @ V_conj)

        gen_to_bus_ei = edge_index_dict[("gen", "connected_to", "bus")]
        target_pg_agg = scatter_add(
            target_dict["gen"][:, PG_H],
            gen_to_bus_ei[1],
            dim=0,
            dim_size=num_bus,
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
            Pd = torch.where(
                mask_bus[:, PD_H],
                pred_bus[:, PD_OUT],
                target_bus[:, PD_H],
            )
            Qd = torch.where(
                mask_bus[:, QD_H],
                pred_bus[:, QD_OUT],
                target_bus[:, QD_H],
            )
        else:
            Pd = target_bus[:, PD_H]
            Qd = target_bus[:, QD_H]
        if pred_bus.size(1) > QG_OUT:
            Qg = torch.where(
                mask_bus[:, QG_H],
                pred_bus[:, QG_OUT],
                target_bus[:, QG_H],
            )
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

    # ------------------------------------------------------------------
    # New path: electrical nodes, slack power from the label,
    # ------------------------------------------------------------------
    @staticmethod
    def _injection(vr, vi, edge_index, edge_attr, num_bus, gs=None, bs=None):
        """Nodal injection S = V * conj(Y V) in real arithmetic."""
        src, dst = edge_index[0], edge_index[1]
        g_ft = edge_attr[:, YFT_TF_R]
        b_ft = edge_attr[:, YFT_TF_I]
        g_d = scatter_add(edge_attr[:, YFF_TT_R], src, dim=0, dim_size=num_bus)
        b_d = scatter_add(edge_attr[:, YFF_TT_I], src, dim=0, dim_size=num_bus)
        if gs is not None:
            g_d = g_d + gs
        if bs is not None:
            b_d = b_d + bs
        i_re = (
            g_d * vr
            - b_d * vi
            + scatter_add(
                g_ft * vr[dst] - b_ft * vi[dst],
                src,
                dim=0,
                dim_size=num_bus,
            )
        )
        i_im = (
            g_d * vi
            + b_d * vr
            + scatter_add(
                g_ft * vi[dst] + b_ft * vr[dst],
                src,
                dim=0,
                dim_size=num_bus,
            )
        )
        p_inj = vr * i_re + vi * i_im
        q_inj = vi * i_re - vr * i_im
        return p_inj, q_inj

    @staticmethod
    @torch.no_grad()
    def electrical_nodes(edge_index, is_switch, num_bus):
        """Electrical node of every bus: connected components over switch edges.

        Min-label propagation with pointer jumping; labels only decrease and
        stay inside a component, so it converges to the smallest bus index of
        each component (in a few iterations: switch groups are small).
        Returns ``(node_of_bus [num_bus], num_nodes)``.
        """
        label = torch.arange(num_bus, device=edge_index.device)
        if num_bus == 0 or not bool(is_switch.any()):
            return label, num_bus
        a, b = edge_index[0, is_switch], edge_index[1, is_switch]
        src = torch.cat([a, b])
        dst = torch.cat([b, a])
        for _ in range(num_bus):
            new = label.scatter_reduce(
                0,
                dst,
                label[src],
                reduce="amin",
                include_self=True,
            )
            new = new[new]
            if torch.equal(new, label):
                break
            label = new
        _, node = torch.unique(label, return_inverse=True)
        return node, int(node.max().item()) + 1

    def switch_edges(self, edge_attr):
        """True for edges whose four admittance entries are (numerically) zero."""
        y = edge_attr[:, [YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I]].abs()
        return (y <= self.switch_tol).all(dim=1)

    @staticmethod
    def _input_bus(model, x_dict, num_bus):
        x_bus = None
        if x_dict is not None:
            x_bus = x_dict["bus"]
        elif model is not None and hasattr(model, "latest_x_dict"):
            x_bus = model.latest_x_dict.get("bus")
        if x_bus is None or x_bus.size(0) != num_bus:
            return None
        return x_bus

    def _forward_nodes(
        self,
        pred_dict,
        target_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        model,
        x_dict,
    ):
        pred_bus = pred_dict["bus"]
        target_bus = target_dict["bus"]
        mask_bus = mask_dict["bus"]
        num_bus = target_bus.size(0)
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]
        x_bus = self._input_bus(model, x_dict, num_bus)

        # ---- voltages: prediction where masked, label elsewhere -------------
        V_m = torch.where(mask_bus[:, VM_H], pred_bus[:, VM_OUT], target_bus[:, VM_H])
        V_a = torch.where(mask_bus[:, VA_H], pred_bus[:, VA_OUT], target_bus[:, VA_H])
        vr = V_m * torch.cos(V_a)
        vi = V_m * torch.sin(V_a)

        gs = bs = None
        if self.include_shunts and x_bus is not None and x_bus.size(1) > BS:
            gs = x_bus[:, GS].to(vr.dtype)
            bs = x_bus[:, BS].to(vr.dtype)
        p_inj, q_inj = self._injection(
            vr,
            vi,
            bus_edge_index,
            bus_edge_attr.to(vr.dtype),
            num_bus,
            gs,
            bs,
        )

        # ---- generation ------------------------------------------------------
        gen_to_bus = edge_index_dict[("gen", "connected_to", "bus")][1]
        any_gen_masked = (
            scatter_add(
                mask_dict["gen"][:, PG_H].to(vr.dtype),
                gen_to_bus,
                dim=0,
                dim_size=num_bus,
            )
            > 0
        )
        if pred_bus.size(1) > QG_OUT:
            Qg = torch.where(
                mask_bus[:, QG_H],
                pred_bus[:, QG_OUT],
                target_bus[:, QG_H],
            )
        else:
            Qg = target_bus[:, QG_H]

        if self.slack_power == "label":
            if target_bus.size(1) <= PG_BUS_H:
                raise ValueError(
                    "PBE slack_power='label' needs the per-bus generation label "
                    f"PG_BUS_H (column {PG_BUS_H}) in the bus targets, which have "
                    f"{target_bus.size(1)} column(s). Use slack_power: gen_rows.",
                )
            ref = self._ref_mask(mask_dict, x_bus, num_bus, target_bus.device)
            Pg = torch.where(
                any_gen_masked & ~ref,
                pred_bus[:, PG_OUT],
                target_bus[:, PG_BUS_H],
            )
            Qg = torch.where(ref, target_bus[:, QG_H], Qg)
        else:
            target_pg_agg = scatter_add(
                target_dict["gen"][:, PG_H],
                gen_to_bus,
                dim=0,
                dim_size=num_bus,
            )
            Pg = torch.where(any_gen_masked, pred_bus[:, PG_OUT], target_pg_agg)

        # Same guard as the legacy path: only outputs with >= 6 columns carry PD/QD.
        if pred_bus.size(1) > BUS_STATUS_TARGET:
            Pd = torch.where(
                mask_bus[:, PD_H],
                pred_bus[:, PD_OUT],
                target_bus[:, PD_H],
            )
            Qd = torch.where(
                mask_bus[:, QD_H],
                pred_bus[:, QD_OUT],
                target_bus[:, QD_H],
            )
        else:
            Pd = target_bus[:, PD_H]
            Qd = target_bus[:, QD_H]

        dP = (Pg - Pd) - p_inj
        dQ = (Qg - Qd) - q_inj

        # ---- electrical nodes (closed switches merged) ----------------------
        if self.merge_switches and bus_edge_index.numel() > 0:
            node, num_nodes = self.electrical_nodes(
                bus_edge_index,
                self.switch_edges(bus_edge_attr),
                num_bus,
            )
            dP_n = scatter_add(dP, node, dim=0, dim_size=num_nodes)
            dQ_n = scatter_add(dQ, node, dim=0, dim_size=num_nodes)
        else:
            node = torch.arange(num_bus, device=dP.device)
            num_nodes = num_bus
            dP_n, dQ_n = dP, dQ
        # |dS| with a finite gradient at an exactly zero residual: the inner
        # where keeps sqrt away from 0, the outer one returns the exact 0.
        sq = dP_n**2 + dQ_n**2
        pos = sq > 0
        mag = torch.where(
            pos,
            torch.sqrt(torch.where(pos, sq, torch.ones_like(sq))),
            torch.zeros_like(sq),
        )

        # ---- scope -----------------------------------------------------------
        if self.bus_scope == "energised":
            if target_bus.size(1) <= BUS_STATUS_TARGET:
                raise ValueError(
                    "PBE bus_scope='energised' needs the bus status label "
                    f"BUS_STATUS_TARGET (column {BUS_STATUS_TARGET}).",
                )
            on_bus = (target_bus[:, BUS_STATUS_TARGET] > 0.5).to(mag.dtype)
            in_scope = (scatter_add(on_bus, node, dim=0, dim_size=num_nodes) > 0).to(
                mag.dtype,
            )
        else:
            in_scope = torch.ones_like(mag)
        n_scope = in_scope.sum().clamp_min(1.0)

        # ---- per-graph load normalisation ------------------------------------
        if self.normalisation == "load":
            weight = self._load_weight(model, target_bus, node, num_nodes, mag.dtype)
            loss = (mag * weight * in_scope).sum() / n_scope
        else:
            loss = (mag * in_scope).sum() / n_scope

        result = {
            "loss": loss,
            "Power loss in p.u.": ((mag * in_scope).sum() / n_scope).detach(),
            "Active Power Loss in p.u.": (
                (dP_n.abs() * in_scope).sum() / n_scope
            ).detach(),
            "Reactive Power Loss in p.u.": (
                (dQ_n.abs() * in_scope).sum() / n_scope
            ).detach(),
        }
        if self.normalisation == "load":
            result["Power loss (load-normalised)"] = loss.detach()
        if self.visualization:
            # per bus (not per node), as before, so the plots keep one value per bus
            result["Nodal Active Power Loss in p.u."] = dP.abs()
            result["Nodal Reactive Power Loss in p.u."] = dQ.abs()
        return result

    @staticmethod
    def _ref_mask(mask_dict, x_bus, num_bus, device):
        if "REF" in mask_dict and mask_dict["REF"] is not None:
            ref = mask_dict["REF"]
            if ref.numel() == num_bus:
                return ref.bool().reshape(-1)
        if x_bus is not None and x_bus.size(1) > REF_H:
            return x_bus[:, REF_H] > 0.5
        raise ValueError(
            "PBE slack_power='label' needs the REF buses: mask_dict['REF'] or the "
            "input REF flag (model.latest_x_dict).",
        )

    @staticmethod
    @torch.no_grad()
    def _load_weight(model, target_bus, node, num_nodes, dtype):
        """1 / (mean |S_load| over the loaded buses of each node's graph)."""
        num_bus = target_bus.size(0)
        device = target_bus.device
        batch = getattr(model, "latest_bus_batch", None) if model is not None else None
        if batch is None or batch.numel() != num_bus:
            batch = torch.zeros(num_bus, dtype=torch.long, device=device)
        batch = batch.to(device=device, dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1 if num_bus > 0 else 1
        s_load = torch.sqrt(target_bus[:, PD_H] ** 2 + target_bus[:, QD_H] ** 2).to(
            dtype,
        )
        loaded = (s_load > 0).to(dtype)
        total = scatter_add(s_load, batch, dim=0, dim_size=num_graphs)
        count = scatter_add(loaded, batch, dim=0, dim_size=num_graphs)
        scale = torch.where(
            count > 0,
            total / count.clamp_min(1.0),
            torch.ones_like(total),
        )
        node_graph = torch.zeros(num_nodes, dtype=torch.long, device=device)
        node_graph.scatter_(0, node, batch)
        return 1.0 / scale[node_graph]


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
