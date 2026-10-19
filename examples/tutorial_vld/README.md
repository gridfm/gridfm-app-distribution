# VLD tutorial (Google Colab)

`Tutorial_vld_colab.ipynb` is a tutorial on the Voltage-Loss-Detection loss
(`VLDTopologyLoss`) ashowing users how to implement it end to end using a small dataset generated from a sample pandapower
distribution network:

1. generate N-1/N-2 contingency data for `case33bw` with `contingency_data_gen`;
2. derive the loss on a five-bus example using the real loss class: BFS
   reachability, class balance $w_{\text{off}}$, the two BCE terms and their
   gradient, the stationary point and the derived threshold
   $\theta = 1/(1+w_{\text{off}})$, the label-wins conflict rule, the voltage terms;
3. tune it: what each $\lambda$ does to the gradients;
4. train a small model with `gridfm_app_dist train` and read the per-term curves;
5. run `gridfm_app_dist predict` with the pre-trained checkpoint in `pretrained/`
   and read the automatic evaluation report.

Open the notebook in Colab (Runtime → T4 GPU), run top to bottom. It installs the application from GitHub, so it always runs against the pushed code.

| file | purpose |
|---|---|
| `Tutorial_vld_colab.ipynb` | the tutorial |
| `tutorial_vld_helpers.py` | CLI runner, MLflow artifact lookup, five-bus toy problem, plots |
| `configs/tutorial_datagen.yaml` | `contingency_data_gen` job: `case33bw`, N-1/N-2 |
| `configs/tutorial_train.yaml` | small VLD-only training config |
| `pretrained/` | trained checkpoint used in step 5 (see its README) |

Run the tutorial from a terminal instead:

```bash
contingency_data_gen --config examples/tutorial_vld/configs/tutorial_datagen.yaml
gridfm_app_dist train --config examples/tutorial_vld/configs/tutorial_train.yaml --data_path data --log_dir mlruns
```
