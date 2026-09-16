"""
Regenerate tests/models/dummy_model.pt.

test_pipeline.py's finetune and evaluate commands load this checkpoint through
main_cli() -> model.load_state_dict(), so its keys and tensor shapes must match
the architecture in tests/config/datamodule_test_base_config.yaml exactly. Run
this whenever model.hidden_size, model.num_layers or model.attention_head change
there, otherwise those two tests fail with a size-mismatch RuntimeError:

    python tests/util/make_dummy_model.py

The weights are random -- nothing asserts anything about the predictions, only
that the load-and-run path works. Keeping the test architecture small keeps this
file small too: the previous 256/8/8 checkpoint was 1.36 GiB of Git LFS that
every CI run had to download.
"""

import os

import torch
import yaml

from gridfm_app_dist.io.param_handler import NestedNamespace, get_task

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(
    REPO_ROOT,
    "tests",
    "config",
    "datamodule_test_base_config.yaml",
)
OUTPUT_PATH = os.path.join(REPO_ROOT, "tests", "models", "dummy_model.pt")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    args = NestedNamespace(**config)
    torch.manual_seed(args.seed)

    # data_normalizers is empty here, exactly as in main_cli(), where the
    # datamodule fills the list after the task is constructed.
    task = get_task(args, [])
    state_dict = task.state_dict()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(state_dict, OUTPUT_PATH)

    params = sum(t.numel() for t in state_dict.values())
    size_mb = os.path.getsize(OUTPUT_PATH) / 2**20
    print(
        f"wrote {OUTPUT_PATH}: {params / 1e6:.2f}M tensor elements, {size_mb:.1f} MiB",
    )


if __name__ == "__main__":
    main()
