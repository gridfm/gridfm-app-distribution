# ============================================================
#          Integrated MCS Reliability Assessment Tool
# ============================================================
#               (c) 2026 Mikka Kisuule | INESC TEC, Porto
# ============================================================
"""Library for building the generic MV/LV pandapower networks.

Typical use::

    from contingency_data_gen.netlib import NetworkBuilder

    builder = NetworkBuilder.from_yaml("contingency_data_gen/configs/netbuilder/urban_MVLV.yaml")
    net = builder.build()          # pandapower net, ready for the MCS model
    builder.export("Results")      # network.json + power flow workbook
"""

from .network_builder import (
    BuildReport,
    NetworkBuilder,
    NetworkConfigError,
    NetworkDiagnosticError,
    build_network,
)
from .std_libraries import LINE_TYPES, STD_TYPES, TRAFO_TYPES

__version__ = "1.0.0"

__all__ = [
    "NetworkBuilder",
    "BuildReport",
    "NetworkConfigError",
    "NetworkDiagnosticError",
    "build_network",
    "STD_TYPES",
    "LINE_TYPES",
    "TRAFO_TYPES",
    "__version__",
]
