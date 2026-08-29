import argparse
import platform
import warnings
from datetime import datetime
from gridfm_app_dist_datakit.cli import main_cli, benchmark_cli

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
    LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split( " ")  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    HOST_LIST = LSB_MCPU_HOSTS[::2]  # Strips the cores per node items in the list
    LSB_JOBID = os.environ["LSB_JOBID"]  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    os.environ["MASTER_ADDR"] = HOST_LIST[0]  # Sets the MasterNode to thefirst node on the list of hosts
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
            raise RuntimeError(f"Unable to compute NODE_RANK from LSF metadata: HOSTNAME='{current_host}', HOST_LIST={HOST_LIST}")

    os.environ["NODE_RANK"] = str(node_rank)  # Uses the list index for node rank, master node rank must be 0
    os.environ["NCCL_SOCKET_IFNAME"] = ("ib,bond")  # avoids using docker of loopback interface
    os.environ["NCCL_IB_CUDA_SUPPORT"] = "1"  # Force use of infiniband


def main():
    parser = argparse.ArgumentParser(prog="gridfm_app_dist_datakit", description="gridfm-app-distribution CLI for the gridfm_app_dist_datakit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ---- GENERATE COMMAND ----
    generator_parser = subparsers.add_parser("generate", help="Run data generation")
    generator_parser.add_argument("--config", type=str, required=True)
    generator_parser.add_argument("--exp_name", type=str, default=exp_name)
    generator_parser.add_argument("--run_name", type=str, default="run")
    generator_parser.add_argument("--log_dir", type=str, default="mlruns")
    generator_parser.add_argument("--data_path", type=str, default="data")


    args = parser.parse_args()
    main_cli(args)

if __name__ == "__main__":
    main()
