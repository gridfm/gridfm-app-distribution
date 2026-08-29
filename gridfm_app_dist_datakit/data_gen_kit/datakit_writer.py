"""
datakit_writer.py
=================

Partitioned-parquet writer matching the gridfm-datakit layout.

Datakit writes each dataset type to its own *directory* of parquet files,
Hive-partitioned by a `scenario_partition` column:

    <raw_dir>/bus_data.parquet/scenario_partition=<k>/*.parquet
    <raw_dir>/gen_data.parquet/scenario_partition=<k>/*.parquet
    <raw_dir>/branch_data.parquet/scenario_partition=<k>/*.parquet

with  The dataloader in gridfm_dist_app (`scenario_partition = scenario // N_SCENARIO_PER_PARTITION`)
reads these with `pd.read_parquet(os.path.join(raw_dir, "bus_data.parquet"))`,
which transparently reads all partitions back as one frame.

Because the generation pipeline runs many parallel worker batches, we append
one parquet *file* per (dataset, partition, batch) instead of rewriting the
whole directory. pyarrow tolerates multiple files per partition directory, and
`pd.read_parquet(dir)` concatenates them, so parallel appends are safe as long
as filenames are unique (we key them by batch id + uuid).
"""

from __future__ import annotations
import os
import uuid
import pandas as pd

# Datakit default; overridable via env for large runs.
N_SCENARIO_PER_PARTITION = int(os.environ.get("N_SCENARIO_PER_PARTITION", "200"))

BUS_DIR = "bus_data.parquet"
GEN_DIR = "gen_data.parquet"
BRANCH_DIR = "branch_data.parquet"


def _add_partition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scenario_partition"] = (df["scenario"] // N_SCENARIO_PER_PARTITION).astype("int64")
    return df


def _append_partitioned(df: pd.DataFrame, dataset_dir: str, batch_id: int) -> None:
    """Write `df` into `dataset_dir`, one file per scenario_partition value.
    Filenames embed the batch id and a unique id so concurrent workers never clash.
    """
    # Skip if the gen dataframe was not created - GridFM model needs the generator.parquet file
    # so ensure it is created empty with the right column names Even if there are gens in the network
    if df is None and df.empty:
        return

    df = _add_partition(df)
    for part_value, part_df in df.groupby("scenario_partition", sort=True):
        part_dir = os.path.join(dataset_dir, f"scenario_partition={int(part_value)}")
        os.makedirs(part_dir, exist_ok=True)
        fname = f"batch_{batch_id:06d}_{uuid.uuid4().hex[:8]}.parquet"
        out_path = os.path.join(part_dir, fname)
        # Drop the partition column from the file body (Hive convention stores
        # it in the directory name); datakit's reader re-derives it from the path.
        part_df.drop(columns=["scenario_partition"]).to_parquet(
            out_path, engine="pyarrow", index=False,
        )


def save_batch_parquet(raw_dir: str,
                       bus_df: pd.DataFrame,
                       gen_df: pd.DataFrame,
                       branch_df: pd.DataFrame,
                       batch_id: int) -> None:
    """Persist one batch's bus/gen/branch frames as partitioned parquet."""
    _append_partitioned(bus_df, os.path.join(raw_dir, BUS_DIR), batch_id)
    _append_partitioned(gen_df, os.path.join(raw_dir, GEN_DIR), batch_id)
    _append_partitioned(branch_df, os.path.join(raw_dir, BRANCH_DIR), batch_id)


def read_dataset(raw_dir: str, which: str) -> pd.DataFrame:
    """Read back a full dataset (`bus`/`gen`/`branch`) across all partitions.

    Returns an empty DataFrame if the dataset directory is absent or holds no
    partition files (e.g. a network with no generators).
    """
    mapping = {"bus": BUS_DIR, "gen": GEN_DIR, "branch": BRANCH_DIR}
    path = os.path.join(raw_dir, mapping[which])
    if not os.path.isdir(path):
        return pd.DataFrame()
    has_files = any(
        f.endswith(".parquet")
        for _, _, files in os.walk(path)
        for f in files
    )
    if not has_files:
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")
