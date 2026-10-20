"""
gridfm_app_dist -- Grid Foundation Model application for distribution networks.

Sub-packages are imported lazily. The command-line entry points
(``gridfm_app_dist.cli`` / ``__main__``) and the task modules import what they
need explicitly, and every registry (models, losses, tasks, transforms,
normalisers) is populated by the module that defines the registered class, so
nothing here has to be imported eagerly. Keeping this file free of heavy
imports means a light-weight tool such as

    python -m gridfm_app_dist.io.vld_evaluation predictions.parquet

runs with pandas / scikit-learn / matplotlib only, without pulling in torch,
Lightning and the model stack (which on a thread-limited login node can exceed
the per-user task limit), and without the runpy "found in sys.modules" warning
that an eager import of the tasks package caused.

"""

from importlib import import_module as _import_module

__all__ = ["datasets", "io", "models", "tasks", "training"]
__version__ = "0.0.1"


def __getattr__(name):
    if name in __all__:
        return _import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
