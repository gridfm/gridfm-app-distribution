"""AC/DC ground-truth metrics helper.

NOTE (VLD integration):
    This module is carried over from gridfm-graphkit unchanged in spirit. It is
    only invoked from the CLI when ``compute_dc_ac_metrics`` is set in the run
    config, and is *not* part of the VoltageLossDetection training/eval path.

    The full graphkit implementation depends on the AC/DC baseline tasks
    (``pf_ac_dc_baseline.py`` / ``opf_ac_dc_baseline.py``) which are outside the
    scope of the distribution-app task port. The signature below is preserved so
    that ``cli.py`` imports cleanly; port the full body from
    ``gridfm_graphkit/tasks/compute_ac_dc_metrics.py`` if you need AC/DC
    baseline metrics in this app.
"""


def compute_ac_dc_metrics(artifacts_dir, raw_dir, grid_name, sn_mva):
    """Compute ground-truth AC/DC metrics for a network.

    Args:
        artifacts_dir: Run artifacts directory (predictions live here).
        raw_dir: Raw parquet data directory for the grid.
        grid_name: Network/case name.
        sn_mva: System base MVA.

    Raises:
        NotImplementedError: This helper is a compatibility stub in
            gridfm-dist-app. See the module docstring.
    """
    raise NotImplementedError(
        "compute_ac_dc_metrics is not ported into gridfm-dist-app. "
        "It is unrelated to the VoltageLossDetection task. Port the body from "
        "gridfm_graphkit/tasks/compute_ac_dc_metrics.py if AC/DC baseline "
        "metrics are required."
    )
