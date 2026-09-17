import torch
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv
from gridfm_app_dist.io.registries import MODELS_REGISTRY
from gridfm_app_dist.io.param_handler import get_physics_decoder
from torch_scatter import scatter_add
from gridfm_app_dist.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
    bound_with_sigmoid,
)
from gridfm_app_dist.datasets.globals import (
    # Bus feature indices
    VM_H,
    VA_H,
    MIN_VM_H,
    MAX_VM_H,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    PG_OUT_GEN,
    ##############
    PHYSICAL_BUS_DIM,
    ##############
    # Generator feature indices
    PG_H,
    MIN_PG,
    MAX_PG,
    # Edge feature indices (topology gating)
    BRANCH_BASE_STATUS_E,
    BRANCH_CONT_E,
)


@MODELS_REGISTRY.register("GNS_heterogeneous")
class GNS_heterogeneous(nn.Module):
    """
    Heterogeneous version of the Transformer-based GNN for buses and generators.
    - Expects node features as dict: x_dict = {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
    - Expects edge_index_dict and edge_attr_dict with keys:
        ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
      (edge_attr only needed for bus-bus currently; other relations can be None)
    - Keeps the physics residual idea but splits it into bus-step and gen-step residuals.

    ------------------------------------------------------------------------
    ARCHITECTURE OPTIONS (all read from ``args.model``, all with defaults that
    reproduce the intended - not the previous - behaviour)
    ------------------------------------------------------------------------
    head_mode : "split" (default) | "concat" | "average"
        How multi-head attention output is combined, and therefore how wide the
        node representation is between layers.

          "split"   out_channels = hidden_size // attention_head, concat=True.
                    Stream width == hidden_size. This is the standard
                    multi-head convention (d_model split across heads) and the
                    only reading under which ``hidden_size`` means what the
                    config says.
          "concat"  out_channels = hidden_size, concat=True.  Stream width
                    == hidden_size * attention_head.  This is the LEGACY
                    behaviour and the reason 256/8/8 built a 365M-parameter
                    model out of a config that reads as 256-wide.
          "average" out_channels = hidden_size, concat=False.  Stream width
                    == hidden_size, but each head still carries full-width
                    internal projections, so it costs ~6x "split".

    norm_position : "pre" (default) | "post"
        "pre"  -> h = h + dropout(act(conv(LayerNorm(h))))     [stable at depth]
        "post" -> h = h + act(LayerNorm(conv(h)))              [LEGACY]

    residual_mode : "always" (default) | "shape_conditional"
        "always"            residual is applied at every layer; a linear
                            projection is inserted only where widths differ.
        "shape_conditional" LEGACY: the residual is silently dropped whenever
                            widths differ (which is exactly layer 0 under
                            head_mode="concat").

    physics_gate : bool (default True)
        Wrap the physics-residual correction in a per-layer LayerScale gate
        initialised to ``physics_gate_init`` and feed the correction MLP a
        squashed (asinh) residual. With gate 0 the physics path starts as an
        exact no-op instead of injecting an unnormalised, rectified vector.

    physics_gate_init : float (default 0.0)

    ffn_ratio : float (default 0.0 = disabled)
        Width multiplier for a per-layer position-wise feed-forward block
        (the second half of a real transformer layer). 0 disables it and keeps
        the parameter count identical to a pure-attention stack.

    topology_gating : "none" (default) | "message_mask"
        "message_mask" drops bus-bus edges whose branch availability
        (BRANCH_BASE_STATUS_E on, BRANCH_CONT_E off) is false before message
        passing and before the branch-flow computation. This makes
        post-contingency reachability representable in ``num_layers`` hops with
        zero extra parameters. It changes model semantics, so it is opt-in.

    LEGACY-EQUIVALENT CONFIGURATION
    -------------------------------
    To reproduce the previous architecture exactly (same parameter shapes, same
    numerics, loads an old state_dict with strict=True):

        model:
          head_mode: concat
          norm_position: post
          residual_mode: shape_conditional
          physics_gate: false
          ffn_ratio: 0.0
          topology_gating: none
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.num_layers = args.model.num_layers
        self.hidden_dim = args.model.hidden_size
        self.input_bus_dim = args.model.input_bus_dim
        self.input_gen_dim = args.model.input_gen_dim
        self.output_bus_dim = args.model.output_bus_dim
        self.output_gen_dim = args.model.output_gen_dim
        self.edge_dim = args.model.edge_dim
        self.heads = args.model.attention_head
        self.task = args.task.task_name
        self.dropout = getattr(args.model, "dropout", 0.0)
        ####################
        self.is_vld_task = self.task == "VoltageLossDetection"
        ####################

        # ---------------- architecture options ----------------------------
        self.head_mode = getattr(args.model, "head_mode", "split")
        self.norm_position = getattr(args.model, "norm_position", "pre")
        self.residual_mode = getattr(args.model, "residual_mode", "always")
        self.use_physics_gate = bool(getattr(args.model, "physics_gate", True))
        self.physics_gate_init = float(getattr(args.model, "physics_gate_init", 0.0))
        self.ffn_ratio = float(getattr(args.model, "ffn_ratio", 0.0))
        self.topology_gating = getattr(args.model, "topology_gating", "none")

        if self.num_layers < 1:
            raise ValueError(
                f"model.num_layers must be >= 1, got {self.num_layers}. The forward "
                "pass decodes inside the layer loop, so zero layers would leave the "
                "bus output undefined.",
            )
        if self.head_mode not in ("split", "concat", "average"):
            raise ValueError(
                f"model.head_mode must be 'split', 'concat' or 'average', "
                f"got {self.head_mode!r}.",
            )
        if self.norm_position not in ("pre", "post"):
            raise ValueError(
                f"model.norm_position must be 'pre' or 'post', got {self.norm_position!r}.",
            )
        if self.residual_mode not in ("always", "shape_conditional"):
            raise ValueError(
                "model.residual_mode must be 'always' or 'shape_conditional', "
                f"got {self.residual_mode!r}.",
            )
        if self.topology_gating not in ("none", "message_mask"):
            raise ValueError(
                "model.topology_gating must be 'none' or 'message_mask', "
                f"got {self.topology_gating!r}.",
            )

        # Per-head output width and the resulting node-representation width.
        # NOTE: HeteroConv does NOT reduce over heads - its `aggr` reduces over
        # RELATIONS. TransformerConv(concat=True) emits heads * out_channels, so
        # out_channels must be hidden_dim // heads for the stream to stay
        # hidden_dim wide. Getting this wrong is a ~64x parameter difference at
        # 256/8/8, not a rounding detail.
        if self.head_mode == "split":
            if self.hidden_dim % self.heads != 0:
                raise ValueError(
                    f"model.hidden_size ({self.hidden_dim}) must be divisible by "
                    f"model.attention_head ({self.heads}) when head_mode='split', "
                    "because each head receives hidden_size // attention_head "
                    "channels and the heads are concatenated back to hidden_size.",
                )
            self.head_dim = self.hidden_dim // self.heads
            self.concat_heads = True
            self.stream_dim = self.hidden_dim
        elif self.head_mode == "concat":
            self.head_dim = self.hidden_dim
            self.concat_heads = True
            self.stream_dim = self.hidden_dim * self.heads
        else:  # "average"
            self.head_dim = self.hidden_dim
            self.concat_heads = False
            self.stream_dim = self.hidden_dim

        # projections for each node type
        self.input_proj_bus = nn.Sequential(
            nn.Linear(self.input_bus_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_gen = nn.Sequential(
            nn.Linear(self.input_gen_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_edge = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # Physics correction MLP: maps the (P, Q) nodal residual into the latent
        # stream. Legacy form is Linear(2, W) + LeakyReLU, i.e. a rectified,
        # unnormalised, non-zero-mean vector added to h_bus at every layer.
        # Gated form ends in a plain Linear (no activation) and is scaled by a
        # per-layer LayerScale parameter so the path starts as identity.
        if self.use_physics_gate:
            self.physics_mlp = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.stream_dim),
            )
            self.physics_scale = nn.Parameter(
                torch.full((self.num_layers,), self.physics_gate_init),
            )
        else:
            self.physics_mlp = nn.Sequential(
                nn.Linear(2, self.stream_dim),
                nn.LeakyReLU(),
            )
            self.physics_scale = None

        # Build hetero layers: HeteroConv of TransformerConv per relation
        self.layers = nn.ModuleList()
        self.norms_bus = nn.ModuleList()
        self.norms_gen = nn.ModuleList()
        self.res_proj_bus = nn.ModuleList()
        self.res_proj_gen = nn.ModuleList()
        self.ffn_bus = nn.ModuleList() if self.ffn_ratio > 0 else None
        self.ffn_gen = nn.ModuleList() if self.ffn_ratio > 0 else None
        self.ffn_norms_bus = nn.ModuleList() if self.ffn_ratio > 0 else None
        self.ffn_norms_gen = nn.ModuleList() if self.ffn_ratio > 0 else None

        for i in range(self.num_layers):
            # Layer 0 consumes the input projections (hidden_dim); deeper layers
            # consume the stream width produced by the previous layer.
            in_bus = self.hidden_dim if i == 0 else self.stream_dim
            in_gen = self.hidden_dim if i == 0 else self.stream_dim
            out_dim = self.head_dim  # PER-HEAD width, see head_mode above

            # relation -> conv module mapping
            conv_dict = {
                ("bus", "connects", "bus"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    concat=self.concat_heads,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("gen", "connected_to", "bus"): TransformerConv(
                    in_gen,
                    out_dim,
                    heads=self.heads,
                    concat=self.concat_heads,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("bus", "connected_to", "gen"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    concat=self.concat_heads,
                    dropout=self.dropout,
                    beta=True,
                ),
            }

            hetero_conv = HeteroConv(conv_dict, aggr="sum")
            self.layers.append(hetero_conv)

            # Norms operate on the stream width. In "pre" position they
            # normalise the layer INPUT; in "post" they normalise the conv
            # OUTPUT (legacy).
            norm_dim_in = in_bus if self.norm_position == "pre" else self.stream_dim
            self.norms_bus.append(nn.LayerNorm(norm_dim_in))
            self.norms_gen.append(nn.LayerNorm(norm_dim_in))

            # Residual path. Identity when widths already agree; a linear
            # projection when they do not (layer 0 under head_mode="concat").
            if self.residual_mode == "always" and in_bus != self.stream_dim:
                self.res_proj_bus.append(nn.Linear(in_bus, self.stream_dim, bias=False))
                self.res_proj_gen.append(nn.Linear(in_gen, self.stream_dim, bias=False))
            else:
                self.res_proj_bus.append(nn.Identity())
                self.res_proj_gen.append(nn.Identity())

            if self.ffn_ratio > 0:
                ffn_hidden = int(round(self.ffn_ratio * self.stream_dim))
                self.ffn_norms_bus.append(nn.LayerNorm(self.stream_dim))
                self.ffn_norms_gen.append(nn.LayerNorm(self.stream_dim))
                self.ffn_bus.append(
                    nn.Sequential(
                        nn.Linear(self.stream_dim, ffn_hidden),
                        nn.LeakyReLU(),
                        nn.Linear(ffn_hidden, self.stream_dim),
                    ),
                )
                self.ffn_gen.append(
                    nn.Sequential(
                        nn.Linear(self.stream_dim, ffn_hidden),
                        nn.LeakyReLU(),
                        nn.Linear(ffn_hidden, self.stream_dim),
                    ),
                )

        # Separate shared MLPs to produce final bus/gen outputs (predictions y)
        self._state_bus_out = VA_H + 1 - VM_H
        self._decoded_bus_out = PHYSICAL_BUS_DIM + (1 if self.is_vld_task else 0)
        if self.output_bus_dim != self._decoded_bus_out:
            raise ValueError(
                f"model.output_bus_dim={self.output_bus_dim} is inconsistent with task "
                f"'{self.task}': the model emits {PHYSICAL_BUS_DIM} physical bus columns"
                + (" plus 1 status logit" if self.is_vld_task else "")
                + f", i.e. {self._decoded_bus_out}.",
            )

        self.mlp_bus = nn.Sequential(
            nn.Linear(self.stream_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self._state_bus_out),
        )

        #############################
        self.bus_status_head = nn.Sequential(
            nn.Linear(self.stream_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        ############################

        self.mlp_gen = nn.Sequential(
            nn.Linear(self.stream_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_gen_dim),
        )

        # In pre-norm the residual stream is never normalised by the layer
        # norms, so it must be normalised once before the decoders.
        if self.norm_position == "pre":
            self.norm_out_bus = nn.LayerNorm(self.stream_dim)
            self.norm_out_gen = nn.LayerNorm(self.stream_dim)
        else:
            self.norm_out_bus = nn.Identity()
            self.norm_out_gen = nn.Identity()

        # mask parameters
        self.activation = nn.LeakyReLU()
        self.res_dropout = nn.Dropout(self.dropout)
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()
        self.node_residuals_layer = ComputeNodeResiduals()
        self.physics_decoder = get_physics_decoder(args)

        # container for monitoring residual norms per layer and type.
        # NOTE: ``layer_residuals`` stays graph-attached on purpose - the
        # registered ``LayeredWeightedPhysics`` loss consumes these tensors as a
        # loss term (see training/loss.py). ``layer_residual_norms`` is the
        # detached copy for logging, so monitoring code never has to keep the
        # autograd graph alive.
        self.layer_residuals = {}
        self.layer_residual_norms = {}

    # ------------------------------------------------------------------
    def _available_branch_mask(self, edge_attr, threshold=0.5):
        """True where a bus-bus branch is in service post-contingency.

        Uses the same two columns and the same threshold convention as
        ``VLDTopologyLoss._build_graph_reachability`` so the model and the
        objective agree on what "available" means.
        """
        base_on = edge_attr[:, BRANCH_BASE_STATUS_E] > threshold
        hit = edge_attr[:, BRANCH_CONT_E] > threshold
        return base_on & (~hit)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        """
        x_dict: {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
        edge_index_dict: keys like ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
        edge_attr_dict: same keys -> edge attributes (bus-bus requires G,B)
        batch_dict: dict mapping node types to batch tensors (if using batching). Not used heavily here but kept for API parity.
        mask: optional mask per node (applies when computing residuals)
        """

        self.layer_residuals = {}
        self.layer_residual_norms = {}
        #####################
        # VLDTopologyLoss reads model.latest_x_dict["bus"] and uses it only for
        # threshold comparisons when building hard reachability labels, so no
        # gradient is required. Detaching keeps a stale autograd graph from
        # being reachable from the module between steps.
        self.latest_x_dict = {key: value.detach() for key, value in x_dict.items()}
        ####################

        # 1) initial projections
        h_bus = self.input_proj_bus(x_dict["bus"])  # [num_bus, hidden_dim]
        h_gen = self.input_proj_gen(x_dict["gen"])  # [num_gen, hidden_dim]

        num_bus = x_dict["bus"].size(0)
        _, gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        # Optional hard topology gate: remove unavailable branches from message
        # passing AND from the branch-flow computation, which is what "the
        # branch is open" physically means. Zero new parameters.
        if self.topology_gating == "message_mask":
            keep = self._available_branch_mask(bus_edge_attr)
            bus_edge_index = bus_edge_index[:, keep]
            bus_edge_attr = bus_edge_attr[keep]
            edge_index_dict = dict(edge_index_dict)
            edge_index_dict[("bus", "connects", "bus")] = bus_edge_index
            edge_attr_dict = dict(edge_attr_dict)
            edge_attr_dict[("bus", "connects", "bus")] = bus_edge_attr

        edge_attr_proj_dict = {}
        for key, edge_attr in edge_attr_dict.items():
            if edge_attr is not None:
                edge_attr_proj_dict[key] = self.input_proj_edge(edge_attr)
            else:
                edge_attr_proj_dict[key] = None

        bus_mask = mask_dict["bus"][:, VM_H : VA_H + 1]
        gen_mask = mask_dict["gen"][:, : (PG_H + 1)]
        bus_fixed = x_dict["bus"][:, VM_H : VA_H + 1]
        gen_fixed = x_dict["gen"][:, : (PG_H + 1)]

        last_layer = self.num_layers - 1
        output_temp = None
        gen_temp = None

        # iterate layers
        for i, conv in enumerate(self.layers):
            if self.norm_position == "pre":
                conv_in_bus = self.norms_bus[i](h_bus)
                conv_in_gen = self.norms_gen[i](h_gen)
            else:
                conv_in_bus, conv_in_gen = h_bus, h_gen

            out_dict = conv(
                {"bus": conv_in_bus, "gen": conv_in_gen},
                edge_index_dict,
                edge_attr_proj_dict,
            )
            out_bus = out_dict["bus"]  # [Nb, stream_dim]
            out_gen = out_dict["gen"]  # [Ng, stream_dim]

            if self.norm_position == "pre":
                # Activation lives INSIDE the block; the residual path stays
                # linear, so the stream magnitude does not drift with depth.
                out_bus = self.res_dropout(self.activation(out_bus))
                out_gen = self.res_dropout(self.activation(out_gen))
            else:
                out_bus = self.activation(self.norms_bus[i](out_bus))
                out_gen = self.activation(self.norms_gen[i](out_gen))

            # skip connection
            if self.residual_mode == "always":
                h_bus = self.res_proj_bus[i](h_bus) + out_bus
                h_gen = self.res_proj_gen[i](h_gen) + out_gen
            else:
                # LEGACY: silently drops the residual whenever widths differ.
                h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
                h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            if self.ffn_ratio > 0:
                h_bus = h_bus + self.res_dropout(
                    self.ffn_bus[i](self.ffn_norms_bus[i](h_bus)),
                )
                h_gen = h_gen + self.res_dropout(
                    self.ffn_gen[i](self.ffn_norms_gen[i](h_gen)),
                )

            # ---- decode ---------------------------------------------------
            # Only the physical state decode has to run every layer, because it
            # feeds the branch-flow / residual computation that the physics
            # feedback consumes. The status and generator heads are read once,
            # after the loop, so calling them per layer was pure waste.
            is_last = i == last_layer

            if self.task == "StateEstimation":
                if not is_last:
                    continue
                bus_temp = self.mlp_bus(self.norm_out_bus(h_bus))
                Pft, Qft = self.branch_flow_layer(
                    bus_temp,
                    bus_edge_index,
                    bus_edge_attr,
                )
                P_in, Q_in = self.node_injection_layer(
                    Pft,
                    Qft,
                    bus_edge_index,
                    num_bus,
                )
                output_temp = self.physics_decoder(
                    P_in,
                    Q_in,
                    bus_temp,
                    x_dict["bus"],
                    None,
                    None,
                )

            else:
                bus_temp = self.mlp_bus(
                    self.norm_out_bus(h_bus) if is_last else h_bus,
                )
                gen_temp = self.mlp_gen(
                    self.norm_out_gen(h_gen) if is_last else h_gen,
                )

                bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
                gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)

                if self.task == "OptimalPowerFlow":
                    # Build the bounded tensor out of place. The previous
                    # in-place column assignment on an autograd-tracked tensor
                    # works only because torch.where produced a fresh tensor,
                    # and breaks as soon as that changes.
                    vm_bounded = bound_with_sigmoid(
                        bus_temp[:, VM_OUT],
                        x_dict["bus"][:, MIN_VM_H],
                        x_dict["bus"][:, MAX_VM_H],
                    )
                    bus_temp = torch.stack(
                        [vm_bounded, bus_temp[:, VA_OUT]],
                        dim=1,
                    )
                    pg_bounded = bound_with_sigmoid(
                        gen_temp[:, PG_OUT_GEN],
                        x_dict["gen"][:, MIN_PG],
                        x_dict["gen"][:, MAX_PG],
                    )
                    gen_temp = torch.cat(
                        [pg_bounded.unsqueeze(1), gen_temp[:, PG_OUT_GEN + 1 :]],
                        dim=1,
                    )

                Pft, Qft = self.branch_flow_layer(
                    bus_temp,
                    bus_edge_index,
                    bus_edge_attr,
                )
                P_in, Q_in = self.node_injection_layer(
                    Pft,
                    Qft,
                    bus_edge_index,
                    num_bus,
                )
                # Reduce gen_temp to a per-generator scalar (the PG output column) WITHOUT collapsing the batch dim.
                # A bare .squeeze() turns a single generator ([1, 1]) into a 0-d scalar,
                # Which scatter_add cannot broadcast against the 1-d gen->bus index
                # Select the PG column and keep dim 0 instead.
                if gen_temp.dim() > 1:
                    gen_contrib = gen_temp[:, PG_OUT_GEN]
                else:
                    gen_contrib = gen_temp.reshape(-1)
                agg_bus = scatter_add(
                    gen_contrib,
                    gen_to_bus_index,
                    dim=0,
                    dim_size=num_bus,
                )

                output_temp = self.physics_decoder(
                    P_in,
                    Q_in,
                    bus_temp,
                    x_dict["bus"],
                    agg_bus,
                    mask_dict,
                )
                residual_P, residual_Q = self.node_residuals_layer(
                    P_in,
                    Q_in,
                    output_temp,
                    x_dict["bus"],
                )

                bus_residuals = torch.stack([residual_P, residual_Q], dim=-1)

                # Save residual norms. `layer_residuals` keeps its graph because
                # LayeredWeightedPhysics uses it as a loss term; the detached
                # copy is what logging should read.
                residual_norm = torch.linalg.norm(bus_residuals, dim=-1).mean()
                self.layer_residuals[i] = residual_norm
                self.layer_residual_norms[i] = residual_norm.detach()

                # The final layer's correction was dead compute: h_bus is never
                # read again after the loop, so that MLP call and its gradients
                # had no effect on the output (verified by ablation).
                if not is_last:
                    correction = self.physics_mlp(
                        torch.asinh(bus_residuals)
                        if self.use_physics_gate
                        else bus_residuals,
                    )
                    if self.use_physics_gate:
                        correction = self.physics_scale[i] * correction
                    h_bus = h_bus + correction

        #######################
        status_logit = self.bus_status_head(self.norm_out_bus(h_bus))

        # return {"bus": output_temp, "gen": gen_temp}
        if self.is_vld_task:
            output_bus = torch.cat([output_temp, status_logit], dim=1)
        else:
            output_bus = output_temp

        if gen_temp is None:  # StateEstimation path does not decode generators
            gen_temp = self.mlp_gen(self.norm_out_gen(h_gen))

        return {"bus": output_bus, "gen": gen_temp}
        #####################
