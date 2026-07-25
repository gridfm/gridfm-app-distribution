import torch
from torch import nn
from torch_scatter import scatter_add

from gridfm_dist_app.datasets.globals import (
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

# NOTE (VLD integration):
# In gridfm-graphkit, the physics decoders (PhysicsDecoderOPF / PF / SE) are
# registered in this module. In gridfm-dist-app they have been moved to
# gridfm_dist_app/tasks/utils.py, where the new PhysicsDecoderVLD lives
# alongside them. Registering them here as well would raise a duplicate-key
# KeyError from PHYSICS_DECODER_REGISTRY, so the registrations are intentionally
# kept only in tasks/utils.py. This module now provides the shared physics
# building blocks only.


class ComputeBranchFlow(nn.Module):
    """Compute sending-end branch flows (Pf, Qf) for all branches."""

    def forward(self, bus_data, edge_index, edge_attr):
        from_idx, to_idx = edge_index

        Vf_mag, Vf_ang = bus_data[from_idx, VM_OUT], bus_data[from_idx, VA_OUT]
        Vt_mag, Vt_ang = bus_data[to_idx, VM_OUT], bus_data[to_idx, VA_OUT]

        Vf_r = Vf_mag * torch.cos(Vf_ang)
        Vf_i = Vf_mag * torch.sin(Vf_ang)
        Vt_r = Vt_mag * torch.cos(Vt_ang)
        Vt_i = Vt_mag * torch.sin(Vt_ang)

        Yfftt_r, Yfftt_i = edge_attr[:, YFF_TT_R], edge_attr[:, YFF_TT_I]
        Yfttf_r, Yfttf_i = edge_attr[:, YFT_TF_R], edge_attr[:, YFT_TF_I]

        Ift_r = Yfftt_r * Vf_r - Yfftt_i * Vf_i + Yfttf_r * Vt_r - Yfttf_i * Vt_i
        Ift_i = Yfftt_r * Vf_i + Yfftt_i * Vf_r + Yfttf_r * Vt_i + Yfttf_i * Vt_r

        Pft = Vf_r * Ift_r + Vf_i * Ift_i
        Qft = Vf_i * Ift_r - Vf_r * Ift_i

        return Pft, Qft


class ComputeNodeInjection(nn.Module):
    """Aggregate branch flows into node-level incoming injections."""

    def forward(self, Pft, Qft, edge_index, num_bus):
        from_idx, _ = edge_index
        P_in = scatter_add(Pft, from_idx, dim=0, dim_size=num_bus)
        Q_in = scatter_add(Qft, from_idx, dim=0, dim_size=num_bus)
        return P_in, Q_in


def compute_shunt_power(bus_data_pred, bus_data_orig):
    """Compute active/reactive shunt power contributions per bus."""
    p_shunt = -bus_data_orig[:, GS] * bus_data_pred[:, VM_OUT] ** 2
    q_shunt = bus_data_orig[:, BS] * bus_data_pred[:, VM_OUT] ** 2
    return p_shunt, q_shunt


class ComputeNodeResiduals(nn.Module):
    """Compute net residuals per bus combining branch flows, generators, loads, and shunts."""

    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig):
        p_shunt, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)
        residual_P = bus_data_pred[:, PG_OUT] - bus_data_orig[:, PD_H] + p_shunt - P_in
        residual_Q = bus_data_pred[:, QG_OUT] - bus_data_orig[:, QD_H] + q_shunt - Q_in
        return residual_P, residual_Q


def bound_with_sigmoid(pred, low, high):
    """Squash unconstrained predictions into [low, high] with a sigmoid map."""
    return low + (high - low) * torch.sigmoid(pred)
