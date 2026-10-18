"""Command line entry point: build one or more networks from YAML input files.

Examples
--------
    python build_network.py configs/urban_MVLV.yaml
    python build_network.py configs/*.yaml --export Results --powerflow
"""

from __future__ import annotations
import argparse
from contingency_data_gen.netlib import NetworkBuilder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", help="YAML input file(s)")
    parser.add_argument(
        "--export",
        metavar="FOLDER",
        help="save network JSON and results to FOLDER",
    )
    parser.add_argument(
        "--powerflow",
        action="store_true",
        help="run a power flow after building",
    )
    parser.add_argument(
        "--no-diagnostic",
        action="store_true",
        help="skip pandapower's diagnostic",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="abort if the diagnostic reports any issue",
    )
    args = parser.parse_args(argv)

    status = 0
    for config in args.configs:
        options = {
            "run_powerflow": args.powerflow,
            "run_diagnostic": not args.no_diagnostic,
            "strict_diagnostic": args.strict,
        }
        builder = NetworkBuilder.from_yaml(config, options=options)
        try:
            builder.build()
        except Exception as exc:
            print(f"[FAIL] {config}: {type(exc).__name__}: {exc}")
            status = 1
            continue

        print(f"[ OK ] {builder.summary()}  ({builder.report.build_seconds:.2f} s)")
        if builder.report.diagnostic:
            print(f"       diagnostic: {sorted(builder.report.diagnostic)}")
        if args.powerflow:
            print(f"       power flow converged: {builder.report.powerflow_converged}")
        if args.export:
            for label, path in builder.export(args.export).items():
                print(f"       {label}: {path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
