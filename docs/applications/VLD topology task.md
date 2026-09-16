## Summary
This features introduces a new Voltage Loss Detection (VLD) task and the associated model changes needed to integrate it.
The goal is to predict post-contingency bus energization status while preserving physically meaningful voltage outputs for buses that remain ON.

This task achieves prediction based load flow simulations in reliability analyses where network contingencies are simulated to identify the customers who are interrupted during each grid failure.
This task also guides the model prediction to be topology-aware by guiding voltage predictions for ON buses towards the reconstructed voltage and towards zero for OFF buses.

The purpose of this task to GridFM users is to enforce topology-awareness in distribution networks during contigency analyses. This task is tailored for distribution networks with multiple failures in MV and LV parts.

---

## Motivation and Context

The existing reconstruction stack is designed to recover physical power-flow quantities, but it does not explicitly model bus de-energization caused by network failures
The VLD task addresses this gap by combining topology-aware supervision with standard physical prediction outputs.
This is useful when the model must distinguish between buses that remain energized and buses that should collapse to an off state after a contingency. It also reduces ambiguity in the training signal by making topology the primary driver of status prediction.

**Possible use-cases**: Reliability assessments, Contigency analysis, Short-circuit analysis, Load flow

**Drivers**: Acceleration of power flow and short-circuit simulations that involve multiple network contingencies.
This is a key research concern as reliability analysis are computationally burdensome when dealing with large network and thousands of contingency scenarios.

**Existing issues, discussions, prototypes**: Ongoing research has focused on optimising MCS-based reliability analysis for implementation on actual networks currently limited by computational requirements.
When combined with MV and LV reliability equivalents, GridFM can enable the accurate reliability assessment of large MV/LV networks using probabilistic methods.

**Background material:** M. Kisuule, et al., "Topology-Aware Graph Transformers for detecting voltage loss after distribution network failures" IEEE Publication (TBC).

---

## Proposed Scope
The intended function of this task is to detect whether a bus survives a contingency while still producing usable physical outputs for the buses that remain active.
The model is trained so that topology determines whether a bus can meaningfully carry voltage.

**Core functionality:**
- Add a VLD-specific task that predicts bus status alongside standard physical outputs.
- Extend the model so the bus head returns physical outputs plus a status logit.
- Use a topology-first loss that treats reachability as the main supervision signal.
- Keep physically reachable buses close to their target Vm values.
- Force unreachable buses toward zero voltage.

**Optional extensions:**
- Add richer calibration metrics for bus status confidence.
- Support alternative topology-confidence weighting strategies.
- Add separate reporting for OFF-bus and ON-bus voltage errors.

---


## Dependencies and Risks

- The task depends on consistent feature indexing across datasets, transforms, losses, and model outputs.
- The topology loss depends on `latest_x_dict` being available during the forward pass.
- Branch pruning must remain consistent with contingency-specific `B_ON` handling.
- The status target layout must match the dataset’s bus target tensor exactly.
- Changes to `grid_path` behavior can accidentally overwrite topology state if not handled carefully.


---

## Testing and Validation Plan

- Unit / integration tests - Unit tests for feature indices and tensor layouts and Integration tests for the VLD transform pipeline.
Loss checks on topology target generation and Vm weighting.
- Benchmarks: Accuracy compared to topology awareness of power flow solvers in reliability assessments
- Reproducibility checks: Evaluation of status accuracy, OFF-bus Vm collapse, and ON-bus Vm retention.
- Real‑world use cases: Distribution reliability assessment

---

## Contribution Plan

- Primary contributor: Kisuule Mikka
- Supervisor: Ignacio Hernando-Gil
- Organization: INESC TEC
- Expected level of maintainer support needed: Medium (available)

---

## Additional Notes

This task can be integrated into the Heterogeneous GridFM code stack to guide the prediction of the model's physical quantities with topology awareness.
