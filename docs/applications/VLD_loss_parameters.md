# VLDTopologyLoss — tuning parameters and how they guide training

This note describes every parameter of `VLDTopologyLoss` as implemented in
`gridfm_app_dist/.../loss.py`, what part of the objective it touches, what moving
it does to the gradients, and how to read the effect on the training curves.
Equation numbers refer to *Topology-Aware Graph Transformers for detecting voltage
loss after distribution network failures* (Kisuule, Bessa, Hernando-Gil).

## 1. What the loss is doing

The model emits, per bus, a status logit $z_i$ and a voltage magnitude $\hat V_i$.
Two labels exist for each bus:

* $y_i^{\text{reach}}$ — derived in the loss by a breadth-first search over
  available buses and branches from the energisation sources (eqs. 1–5).
* $y_i^{\text{target}}$ — the dataset's post-contingency energised flag from the
  power-flow solver.

Four terms are combined (eq. 18):

$$\mathcal{L}_{\text{VLD}} = \lambda_{\text{reach}}\mathcal{L}_{\text{reach}} + \lambda_{\text{anc}}\mathcal{L}_{\text{anc}} + \lambda_{\text{off}}\mathcal{L}_{\text{off}} + \lambda_{\text{on}}\mathcal{L}_{\text{on}}$$

| Term | Supervises | Target | Per-bus weight | Normaliser |
|---|---|---|---|---|
| $\mathcal{L}_{\text{reach}}$ (13) | $z_i$ | $y_i^{\text{reach}}$ | $\omega_i = \kappa_i\,\omega_i^{\text{cls}}$; **0 on conflict buses** | $\sum_i \omega_i$ |
| $\mathcal{L}_{\text{anc}}$ (14) | $z_i$ | $y_i^{\text{target}}$ | $\nu_i$; **$w_{\text{off}}$ on conflict buses** | $\sum_i \nu_i$ |
| $\mathcal{L}_{\text{off}}$ (15) | $\hat V_i \to 0$ | buses with $y_i^{\text{reach}} = 0$ | 1 | $n_{\text{off}}$ |
| $\mathcal{L}_{\text{on}}$ (17) | $\hat V_i \to V_i$ | buses with $y_i^{\text{reach}} = 1$ | $\eta_i = \mathrm{sg}(\hat p_i)$ | $\sum_i \eta_i$ |

Each term is a *weighted mean*, so the $\lambda$ values compare like-for-like
quantities and the batch size does not change their balance.

Two things are fixed by the formulation and are **not** parameters:

* **Conflict rule.** On $\mathcal{C} = \{i : y_i^{\text{reach}} \ne y_i^{\text{target}}\}$
  the topology weight is zero and the anchor weight is $w_{\text{off}}$, so the
  stationary point of $z_i$ is the label, whether 0 or 1. The voltage terms stay
  keyed on $y_i^{\text{reach}}$.
* **Decision threshold.** $\theta = 1/(1 + w_{\text{off}})$. Weighting the 0-class
  by $w$ moves the BCE optimum from the true posterior $q$ to $q/(q + w(1-q))$;
  thresholding at $1/(1+w)$ maps it back to the boundary $q = 0.5$. The value is
  returned as `VLD Decision Threshold` and **must be reused at inference**
  (read $w_{\text{off}}$ from the `ema_off_weight` buffer, which is checkpointed).

## 2. Gradient picture

For the status logit (eq. 19):

$$\frac{\partial \mathcal{L}_{\text{VLD}}}{\partial z_i} = a_i(\hat p_i - y_i^{\text{reach}}) + b_i(\hat p_i - y_i^{\text{target}}),\qquad a_i = \lambda_{\text{reach}}\frac{\omega_i}{\sum_j \omega_j},\quad b_i = \lambda_{\text{anc}}\frac{\nu_i}{\sum_j \nu_j}$$

* Agreeing bus, $y = 1$: slope $a_i + b_i$ with $\omega_i = \nu_i = 1$.
* Agreeing bus, $y = 0$: slope $a_i + b_i$ with $\omega_i = \nu_i = w_{\text{off}}$,
  i.e. $w_{\text{off}}$ times steeper. This is the class balance.
* Conflict bus: slope $b_i$ only, with $\nu_i = w_{\text{off}}$; zero at $y_i^{\text{target}}$.

For the voltage head (eqs. 24–25):

$$\frac{\partial \mathcal{L}_{\text{VLD}}}{\partial \hat V_i} = \begin{cases} \dfrac{\lambda_{\text{off}}}{n_{\text{off}}}\big(2\hat V_i + \lambda_1\,\mathrm{sgn}\,\hat V_i\big) & y_i^{\text{reach}} = 0 \\[1.2em] \dfrac{2\lambda_{\text{on}}\,\mathrm{sg}(\hat p_i)}{\sum_j \eta_j}\big(\hat V_i - V_i\big) & y_i^{\text{reach}} = 1 \end{cases}$$

Neither voltage term reaches $z_i$: $\eta_i$ is a stop-gradient. The status head is
trained only by the two BCE terms.

## 3. Parameters

### 3.1 Term weights — the $\lambda$ of eq. (18)

**`topology_weight`** ($\lambda_{\text{reach}}$, default 1.0). Scale of the primary
BCE. Acts as the reference the other three are expressed against; leave at 1.0 and
tune the others.

**`target_anchor_weight`** ($\lambda_{\text{anc}}$, default 0.5, paper 0.25). Scale
of the label BCE. On agreeing buses it only changes how fast $z_i$ moves (the
stationary point is $y$ regardless). On conflict buses it is the *only* force on
$z_i$, so it sets how quickly label-only buses are learned relative to agreeing
ones. Raised from 0.25 because at 0.25 conflict buses learned four times slower
than the rest. Raise further if `VLD Conflict Rate` is high and
`VLD Target Accuracy` lags `VLD Topology Accuracy`; lower if the BFS label is
known to be more trustworthy than the dataset flag.

**`off_vm_weight`** ($\lambda_{\text{off}}$, default 1.0). Pull of $\hat V_i$
toward 0 on BFS-de-energised buses. Increase if the MAE of $\hat V$ on
de-energised buses plateaus above a few $10^{-2}$ p.u.; decrease if energised-bus
voltage error (`VLD On Vm Loss`) deteriorates, which indicates the shared trunk is
being over-committed to the outage case.

**`on_vm_weight`** ($\lambda_{\text{on}}$, default 0.5). Pull of $\hat V_i$ toward
the solver value on energised buses, weighted by the model's current confidence
$\hat p_i$. Because of that weighting it is weak early (when $\hat p$ is
uncertain) and strengthens as the classifier converges — a built-in curriculum.
Keep below $\lambda_{\text{off}}$; energised buses are the majority and already
receive voltage supervision from the MaskedMSE/PBE terms outside this loss.

**`unreachable_l1_weight`** ($\lambda_1$, default 1.0). Adds a *constant* pull of
magnitude $\lambda_1\lambda_{\text{off}}/n_{\text{off}}$ toward $\hat V = 0$ on
de-energised buses (eq. 24). The L2 part alone decays as $\hat V \to 0$ and
leaves a residual; the L1 part closes it. 1.0 means the constant pull equals the
L2 pull at $\hat V = 0.5$ p.u. Lower to 0.25–0.5 if de-energised voltages
oscillate around zero (sign flips in the gradient at exactly 0).

### 3.2 Class balance — $w_{\text{off}}$ of eq. (6)

**`unreachable_pos_weight`** (default `auto`). `auto` sets
$w_{\text{off}} = n_{\text{on}}/n_{\text{off}}$ from the BFS labels, so the
de-energised minority contributes as much total gradient as the energised
majority. A number fixes $w_{\text{off}}$ (1.0 disables balancing). Use a
number only when the outage rate is known and stable; `auto` adapts to each
dataset and, with momentum, to each training phase.

**`max_pos_weight`** (default 10.0, previously 50.0). Upper clamp on
$w_{\text{off}}$. On the 33-bus N-3 set about 35–40 % of buses are de-energised,
so $w_{\text{off}} \approx 1.6$ and the clamp never binds. It protects sparse
outage sets: at 50, a batch with one de-energised bus in 500 hands that bus ~9 %
of the whole gradient and drives $\theta$ to 0.02. Raise toward 20–30 only for
large meshed networks where outages are genuinely rare, and watch
`VLD On Recall`.

**`pos_weight_momentum`** (default 0.99, 0.0 = per-batch). EMA momentum for the
`auto` weight; 0.99 is roughly a 100-batch window. Because $\theta$ is derived
from $w_{\text{off}}$, a per-batch estimate would make the decision boundary jump
between batches. The EMA is updated only in training mode and frozen for
validation/test, and is saved with the checkpoint. Set to 0.0 to reproduce the
original per-batch behaviour.

### 3.3 Reachability — eqs. (1)–(5)

**`reachability_sources`** (default `ref`). Which buses seed the BFS. `ref` =
slack/ext_grid only, correct for IEEE-1547-style DG that trips on loss of mains
(the 33-bus feeder). `ref_and_gen` also seeds from PV buses, i.e. treats islands
with local generation as energised; only for grid-forming DG or intentional
islanding studies. A wrong choice shows up directly as a high
`VLD Topo-vs-Target Disagreement`.

**`input_state_threshold`** ($\tau$, default 0.5). Threshold applied to the 0/1
status and contingency features. Leave at 0.5 unless the input pipeline rescales
those columns (then the BFS will silently mislabel everything — check
`VLD Off Rate (labels)` against the known outage rate on the first batch).

**`topology_confidence_gamma`** ($\gamma$ of eq. 8, default 0.0). Down-weights
the topology BCE on buses that are available but islanded
($\alpha_i = 1,\ y_i^{\text{reach}} = 0$) by $e^{-\gamma}$. With correct
labels these are exactly the buses VLD exists to detect, so down-weighting them
works against the objective; 0 disables and is recommended. Kept for ablations.

## 4. Reading the training curves

| Metric | What it tells you | Healthy sign |
|---|---|---|
| `VLD Off Recall` / `VLD Off Precision` / `VLD Off F1` | Detection of de-energised buses (the task) | F1 rising, precision not collapsing |
| `VLD On Recall` | $1 -$ false-de-energisation rate | $\ge 0.97$ on the 33-bus set |
| `VLD Balanced Accuracy` | Mean of the two recalls; use for model selection | Rising monotonically |
| `VLD Mean Prob (reachable & label 1)` | Calibration on unambiguous energised buses | Near 1. If it sits near $1/(1+w_{\text{off}})$, the class weight is the cause of a de-energisation bias |
| `VLD Decision Threshold` | $1/(1+w_{\text{off}})$ in use | Stable after the first ~100 batches |
| `VLD Off Class Weight` | $w_{\text{off}}$ (EMA) | $\approx$ true on/off ratio, not pinned at the clamp |
| `VLD Topo-vs-Target Disagreement` | Fraction of conflict buses | Low (< few %); persistent high values mean the BFS or the flags are wrong |
| `VLD Conflict Rate (reach 0, label 1)` / `(reach 1, label 0)` | Direction of the disagreement | The first is the case that used to produce false de-energisations |
| `VLD Off Vm Loss` / `VLD On Vm Loss` | Voltage head on each class | Both decreasing; `On` lags `Off` early by design |

Rule of thumb for a de-energisation bias (`On Recall` low, `Mean Prob` low):
check `Off Class Weight` first (clamp binding? lower `max_pos_weight`), then
`Conflict Rate (reach 0, label 1)` (BFS missing paths? check
`reachability_sources`), then `target_anchor_weight`.

## 5. Suggested sweep

The formulation fixes most of the behaviour; the two parameters that encode a
real trade-off are

* `target_anchor_weight` ∈ {0.25, 0.5, 1.0}
* `on_vm_weight` ∈ {0.25, 0.5}

Select on validation `VLD Balanced Accuracy`, not `VLD Off F1` (F1 on the off
class is blind to false de-energisations). Six runs cover the grid.

## 6. Interaction with the other losses

`loss_weights: [0.05, 0.15, 0.80]` for `MaskedMSE`, `PBE`, `VLDTopologyLoss`.
PBE compares the injection implied by $\hat V$ with the dataset's $P_d, Q_d$. If
the scenario generator leaves pre-contingency load on de-energised buses, PBE
pulls $\hat V$ up on exactly the buses $\mathcal{L}_{\text{off}}$ pulls down.
Either confirm loads are zeroed on de-energised buses in the data, mask PBE on
those buses, or lower its weight.
