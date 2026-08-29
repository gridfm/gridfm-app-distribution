import argparse
import platform
import warnings
from datetime import datetime
from gridfm_app_dist.cli import main_cli, benchmark_cli

import subprocess
import os

def _warn_mp_context_on_linux(mp_context):
    """On Linux, recommend 'spawn' when mp_context is unset, 'fork', or 'forkserver'."""
    if platform.system() != "Linux":
        return
    if mp_context in (None, "fork", "forkserver"):
        chosen = mp_context if mp_context is not None else "PyTorch default"
        warnings.warn(
            f"--mp_context is '{chosen}' on Linux. 'spawn' is recommended for safety "
            "(avoids issues with CUDA initialization and forked processes), though "
            "'fork'/'forkserver' may be faster.",
            stacklevel=2,
        )


def is_lsf():
    return (
        os.environ.get("LSB_JOBID") is not None
        and os.environ.get("LSB_MCPU_HOSTS") is not None
        and "LSF_ENVDIR" in os.environ  # strong LSF indicator
    )


def fix_infiniband():
    """Configure NCCL to skip Ethernet-only IB ports on this host."""
    ibv = subprocess.run("ibv_devinfo", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode("utf-8").split("\n")
    exclude = ""
    for line in lines:
        if "hca_id:" in line:
            name = line.split(":")[1].strip()
        if "\tport:" in line:
            port = line.split(":")[1].strip()
        if "link_layer:" in line and "Ethernet" in line:
            exclude = exclude + f"{name}:{port},"

    if exclude:
        exclude = "^" + exclude[:-1]
        os.environ["NCCL_IB_HCA"] = exclude


def set_env():
    """Populate distributed-training environment variables from LSF metadata."""
    # print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    LSB_MCPU_HOSTS = os.environ[
        "LSB_MCPU_HOSTS"
    ].split(
        " ",
    )  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    HOST_LIST = LSB_MCPU_HOSTS[::2]  # Strips the cores per node items in the list
    LSB_JOBID = os.environ[
        "LSB_JOBID"
    ]  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    os.environ["MASTER_ADDR"] = HOST_LIST[
        0
    ]  # Sets the MasterNode to thefirst node on the list of hosts
    os.environ["MASTER_PORT"] = "5" + LSB_JOBID[-5:-1]
    current_host = os.environ.get("HOSTNAME", "")
    host_list_norm = [h.strip().lower() for h in HOST_LIST]
    current_norm = current_host.strip().lower()

    # LSF host lists can be short names while HOSTNAME may be FQDN.
    if current_norm in host_list_norm:
        node_rank = host_list_norm.index(current_norm)
    else:
        current_short = current_norm.split(".")[0]
        host_list_short = [h.split(".")[0] for h in host_list_norm]
        if current_short in host_list_short:
            node_rank = host_list_short.index(current_short)
        else:
            raise RuntimeError(
                "Unable to compute NODE_RANK from LSF metadata: "
                f"HOSTNAME='{current_host}', HOST_LIST={HOST_LIST}"
            )

    os.environ["NODE_RANK"] = str(
        node_rank,
    )  # Uses the list index for node rank, master node rank must be 0
    os.environ["NCCL_SOCKET_IFNAME"] = (
        "ib,bond"  # avoids using docker of loopback interface
    )
    os.environ["NCCL_IB_CUDA_SUPPORT"] = "1"  # Force use of infiniband



def main():
    parser = argparse.ArgumentParser(
        prog="gridfm_app_dist",
        description="gridfm-app-distribution CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    _compile_kwargs = dict(
        type=str,
        default=None,
        nargs="?",
        const="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="Enable torch.compile with the given mode (omit value for 'default').",
    )
    _bfloat16_kwargs = dict(
        action="store_true",
        default=False,
        help="Cast model to bfloat16 (model.to(torch.bfloat16)).",
    )
    _tf32_kwargs = dict(
        action="store_true",
        default=False,
        help="Enable TF32 on Ampere+ GPUs via torch.set_float32_matmul_precision('high').",
    )

    # ---- TRAIN SUBCOMMAND ----
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--exp_name", type=str, default=exp_name)
    train_parser.add_argument("--run_name", type=str, default="run")
    train_parser.add_argument("--log_dir", type=str, default="mlruns")
    train_parser.add_argument("--data_path", type=str, default="data")
    train_parser.add_argument("--compile", **_compile_kwargs)
    train_parser.add_argument("--bfloat16", **_bfloat16_kwargs)
    train_parser.add_argument("--tf32", **_tf32_kwargs)
    train_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    train_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration, e.g. gridfm_app_dist_ee",
    )
    train_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config. Use 0 to debug worker crashes.",
    )
    train_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache. If set, cache is loaded from here when present and saved here after first population.",
    )
    train_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Enable Lightning profiler: 'simple', 'advanced', or 'pytorch'.",
    )

    # ---- FINETUNE SUBCOMMAND ----
    finetune_parser = subparsers.add_parser("finetune", help="Run fine-tuning")
    finetune_parser.add_argument("--config", type=str, required=True)
    finetune_parser.add_argument("--model_path", type=str, required=True)
    finetune_parser.add_argument("--exp_name", type=str, default=exp_name)
    finetune_parser.add_argument("--run_name", type=str, default="run")
    finetune_parser.add_argument("--log_dir", type=str, default="mlruns")
    finetune_parser.add_argument("--data_path", type=str, default="data")
    finetune_parser.add_argument("--compile", **_compile_kwargs)
    finetune_parser.add_argument("--bfloat16", **_bfloat16_kwargs)
    finetune_parser.add_argument("--tf32", **_tf32_kwargs)
    finetune_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    finetune_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration, e.g. gridfm_app_dist_ee",
    )
    finetune_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config. Use 0 to debug worker crashes.",
    )
    finetune_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache. If set, cache is loaded from here when present and saved here after first population.",
    )
    finetune_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Enable Lightning profiler: 'simple', 'advanced', or 'pytorch'.",
    )

    # ---- EVALUATE SUBCOMMAND ----
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model performance",
    )
    evaluate_parser.add_argument("--model_path", type=str, default=None)
    evaluate_parser.add_argument(
        "--normalizer_stats",
        type=str,
        default=None,
        help="Path to normalizer_stats.pt from a training run.",
    )
    evaluate_parser.add_argument("--config", type=str, required=True)
    evaluate_parser.add_argument("--exp_name", type=str, default=exp_name)
    evaluate_parser.add_argument("--run_name", type=str, default="run")
    evaluate_parser.add_argument("--log_dir", type=str, default="mlruns")
    evaluate_parser.add_argument("--data_path", type=str, default="data")
    evaluate_parser.add_argument("--compile", **_compile_kwargs)
    evaluate_parser.add_argument("--bfloat16", **_bfloat16_kwargs)
    evaluate_parser.add_argument("--tf32", **_tf32_kwargs)
    evaluate_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    evaluate_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration, e.g. gridfm_app_dist_ee",
    )
    evaluate_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config. Use 0 to debug worker crashes.",
    )
    evaluate_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache. If set, cache is loaded from here when present and saved here after first population.",
    )
    evaluate_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Enable Lightning profiler: 'simple', 'advanced', or 'pytorch'.",
    )
    evaluate_parser.add_argument(
        "--compute_dc_ac_metrics",
        action="store_true",
    )
    evaluate_parser.add_argument(
        "--save_output",
        action="store_true",
    )

    # ---- PREDICT SUBCOMMAND ----
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--model_path", type=str, required=False)
    predict_parser.add_argument("--normalizer_stats", type=str, default=None)
    predict_parser.add_argument("--config", type=str, required=True)
    predict_parser.add_argument("--exp_name", type=str, default=exp_name)
    predict_parser.add_argument("--run_name", type=str, default="run")
    predict_parser.add_argument("--log_dir", type=str, default="mlruns")
    predict_parser.add_argument("--data_path", type=str, default="data")
    predict_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    predict_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration, e.g. gridfm_app_dist_ee",
    )
    predict_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config. Use 0 to debug worker crashes.",
    )
    predict_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache. If set, cache is loaded from here when present and saved here after first population.",
    )
    predict_parser.add_argument("--output_path", type=str, default="data")
    predict_parser.add_argument("--compile", **_compile_kwargs)
    predict_parser.add_argument("--bfloat16", **_bfloat16_kwargs)
    predict_parser.add_argument("--tf32", **_tf32_kwargs)
    predict_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
    )

    # ---- BENCHMARK SUBCOMMAND ----
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark train-dataloader iteration speed",
    )
    benchmark_parser.add_argument("--config", type=str, required=True)
    benchmark_parser.add_argument("--data_path", type=str, default="data")
    benchmark_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to iterate through the train dataloader.",
    )
    benchmark_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    benchmark_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache.",
    )
    benchmark_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config.",
    )
    benchmark_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration.",
    )

    args = parser.parse_args()

    if args.command == "benchmark":
        benchmark_cli(args)
    else:
        main_cli(args)


if __name__ == "__main__":
    main()
