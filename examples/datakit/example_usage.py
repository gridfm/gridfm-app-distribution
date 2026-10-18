"""Minimal usage examples for the NetworkBuilder class."""

from __future__ import annotations

from pathlib import Path


from contingency_data_gen.netlib import NetworkBuilder, build_network

CONFIGS = Path(__file__).resolve().parent / "configs"

# 1) simplest form -- one call, net returned
net = build_network(CONFIGS / "urban_LV.yaml")
print(net)

# 2) full control through the builder object
builder = NetworkBuilder.from_yaml(
    CONFIGS / "urban_MVLV.yaml",
    options={"run_diagnostic": True, "run_powerflow": True, "strict_diagnostic": False},
)
net = builder.build()
print(builder.summary())
print("diagnostic issues:", sorted(builder.report.diagnostic) or "none")
print("min bus voltage (pu):", round(net.res_bus.vm_pu.min(), 4))

# bus bookkeeping is available for the reliability model
print("MV feeder subsections:", list(builder.mv_buses["MV_11kV"]["feeder"]))
print("LV networks built:", len(builder.lv_buses))

# 3) override input data on the fly (e.g. sensitivity study on the ADMD)
builder = NetworkBuilder.from_yaml(
    CONFIGS / "urban_LV.yaml",
    load_points={"lp_per_customer": 0.0035, "power_factor": 0.95},
    options={"run_powerflow": True},
)
net = builder.build()
print("peak demand (MW):", round(net.load.p_mw.sum(), 4))

# 4) export network JSON, a build report and the power flow workbook
print(builder.export("Results"))
