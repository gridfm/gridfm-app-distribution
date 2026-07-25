---
name: 🚀 Topology-based Voltage Loss Detection Task
about: Specialised task to predict the bus ON/OFF status following a contingency and guide voltages at failed buses to zero
title: "[Voltage loss prediction and network topology reconstruction] <Topology-based Voltage Loss Detection Task>"
labels: ["VLD-loss"]
assignees: "Mikka Kisuule "
---

## Summary
This features introduces a new Voltage Loss Detection (VLD) task and the associated model changes needed to integrate it.
The goal is to predict post-contingency bus energization status while preserving physically meaningful voltage outputs for buses that remain ON.

This task achieves prediction based load flow simulations in reliability analyses where network contingencies are simulated to identify the customers who are interrupted during each grid failure. 
This task also guides the model prediction to be topology-aware by guiding voltage predictions for ON buses towards the reconstructed voltage and towards zero for OFF buses.

The purpose of this task to GridFM users is to enforce topology-awareness during contingency analyses. 

---

## Motivation and Context
This task aims to train the foundation model on the topographical reconstruction of a network following a grid failure. 

The task classifies buses that are de-energized (lose voltage) as a result of topographical disconnection (due to ag grid failure) from the reference slack bus(es). 


The reconstruction task explicitly models the bus de-energization caused by network failures by combining topology-aware supervision with standard physical prediction outputs.
This is useful when the model must distinguish between buses that remain energized and buses that should collapse to an off state after a contingency. 
It also reduces ambiguity in the training signal by making topology the primary driver of bus status prediction.

**Possible use-cases**: Reliability assessments, Contingency analysis, Power quality assessment.

**Drivers**: Acceleration of power flow analyses that involve simulation of the network with multiple network contingencies (N-k). 
This is a key research concern as reliability analysis are computationally burdensome when dealing with large network and thousands of contingency scenarios. 

**Existing issues, discussions, prototypes**: Ongoing research has focused on optimising MCS-based reliability analysis for implementation on actual networks currently limited by computational requirements. 
When combined with MV and LV reliability equivalents, GridFM can enable the accurate reliability assessment of large MV/LV networks using probabilistic methods. 

**Link any relevant background material:** Ndawula, Mike Brian, et al. "A novel formulation of low voltage distribution network equivalents for reliability analysis." Sustainable Energy, Grids and Networks 39 (2024): 101437.

---

## Detailed Scope
The intended application of this task is to detect whether a bus survives a contingency while still producing usable physical outputs for the buses that remain active. 
The model is trained so that topology determines whether a bus can meaningfully carry voltage. 

**Core functionality:**
- Add a VLD-specific task that predicts bus status alongside standard physical outputs.
- Extend the model so the bus head returns physical outputs plus a status logit.
- Use a topology-first loss that treats reachability as the main supervision signal.
- Keep physically reachable buses close to their target Vm values.
- Force unreachable buses toward zero voltage.

Given a base grid topology and a set of component failures (contingencies) on
buses and branches, the model predicts — for every bus — whether it remains
**energized** after the failures, and a consistent voltage magnitude for the
energized buses. The task encodes the physical rule:

> A bus can only be energized if a path of energized buses and branches connects
> it to a slack bus.

`_build_graph_reachability` in the loss function implements the stated rule as
  an iterative BFS: seed with buses whose base
  status is on and that are not hit by a contingency, then propagate along
  branches that are themselves on and un-hit until the reachable set stops
  growing. This is a fully differentiable encoding of the
  connectivity constraint; the reachability labels are hard; the learned status
  logit is trained against them with a confidence-weighted BCE.

This is learned with a topology-aware objective (`VLDTopologyLoss`) built on top
of the existing heterogeneous GNS reconstruction backbone.


**Optional extensions:**
- Add richer calibration metrics for bus status confidence.
- Support alternative topology-confidence weighting strategies.
- Add separate reporting for OFF-bus and ON-bus voltage errors.

## Maturity Level

How mature is this feature today?

- [ ] Idea / design discussion
- [ ] Research prototype
- [X] Partially implemented
- [ ] Production‑ready

If a prototype exists, please link to code, papers, or branches.

---

## Dependencies and Risks


- The task depends on consistent feature indexing across datasets, transforms, losses, and model outputs.
- The topology loss depends on `latest_x_dict` being available during the forward pass.
- Branch pruning must remain consistent with contingency-specific `B_ON` handling.
- The status target layout must match the dataset’s bus target tensor exactly.
- Changes to `grid_path` behavior can accidentally overwrite topology state if not handled carefully.


## Testing and Validation Plan

- Unit / integration tests - Unit tests for feature indices and tensor layouts and Integration tests for the VLD transform pipeline. 
Loss checks on topology target generation and Vm weighting. 
- Benchmarks: Accuracy compared to topology awareness of power flow solvers in reliability assessments
- Reproducibility checks: Evaluation of status accuracy, OFF-bus Vm collapse, and ON-bus Vm retention.
- Real‑world use cases: Distribution reliability assessment 

---

## Contribution Plan

- Primary contributor(s): Kisuule Mikka 
- Supervisor: Ignacio Hernando-Gil
- Organization: INESC TEC
- Expected level of maintainer support needed: High (available)

---

## Additional Notes

This task can be integrated into the Heterogeneous GridFM code stack to guide the prediction of the model's physical quantities with topology awareness. 