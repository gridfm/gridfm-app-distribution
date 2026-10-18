"""
contingency_data_gen -- scenario / contingency data generation for gridfm_app_dist.

Builds pandapower networks (from Excel workbooks, YAML network definitions,
MATPOWER files or pandapower's built-in cases), runs N-k contingency analyses
and writes the results as gridfm-datakit-format partitioned parquet under
``<output_dir>/raw/``, which is exactly the layout ``gridfm_app_dist`` reads
from ``--data_path``.

Command line::

    contingency_data_gen --config contingency_data_gen/configs/data_gen_1.yaml
    contingency_data_gen --list-configs

Python::

    from contingency_data_gen.generation.scenario_data_gen import ScenarioGenerator
    from contingency_data_gen.netlib import NetworkBuilder
"""

from importlib.resources import files as _files

__version__ = "0.0.1"

CONFIG_DIR = _files("contingency_data_gen") / "configs"

__all__ = ["CONFIG_DIR", "__version__"]
