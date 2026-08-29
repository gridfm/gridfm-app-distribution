# Package rename report

## Summary

An isolated copy of the project was created at:

`C:\Users\mikka.kisuule\OneDrive - INESC TEC\Documents\New project\gridfm-app-distribution`

The Python package and command identifier were renamed from `gridfm_dist_app` to
`gridfm_app_dist`. The original project at
`C:\Temp\gridfm-app-distribution\gridfm-app-distribution` was not modified.

## Changes made

- Renamed the package directory from `gridfm_dist_app/` to `gridfm_app_dist/`.
- Updated Python imports and module references throughout the application.
- Updated the `python -m` module target and displayed CLI program name.
- Updated the setuptools package-discovery pattern and console-script entry point.
- Updated unit-test imports, command arguments, and test descriptions.
- Updated README command examples and application documentation.
- Updated the checked-in `.egg-info` package metadata and source manifest.

## Files containing identifier changes

### Packaging and generated metadata

- `pyproject.toml`
- `gridfm_app_distribution.egg-info/PKG-INFO`
- `gridfm_app_distribution.egg-info/SOURCES.txt`
- `gridfm_app_distribution.egg-info/entry_points.txt`
- `gridfm_app_distribution.egg-info/top_level.txt`

### Application package

- `gridfm_app_dist/__init__.py`
- `gridfm_app_dist/__main__.py`
- `gridfm_app_dist/cli.py`
- `gridfm_app_dist/datasets/__init__.py`
- `gridfm_app_dist/datasets/hetero_powergrid_datamodule.py`
- `gridfm_app_dist/datasets/masking.py`
- `gridfm_app_dist/datasets/normalizers.py`
- `gridfm_app_dist/datasets/powergrid_hetero_dataset.py`
- `gridfm_app_dist/datasets/task_transforms.py`
- `gridfm_app_dist/datasets/transforms.py`
- `gridfm_app_dist/io/param_handler.py`
- `gridfm_app_dist/io/visualization.py`
- `gridfm_app_dist/models/__init__.py`
- `gridfm_app_dist/models/gnn_heterogeneous_gns.py`
- `gridfm_app_dist/models/utils.py`
- `gridfm_app_dist/tasks/__init__.py`
- `gridfm_app_dist/tasks/opf_task.py`
- `gridfm_app_dist/tasks/pf_task.py`
- `gridfm_app_dist/tasks/reconstruction_tasks.py`
- `gridfm_app_dist/tasks/se_task.py`
- `gridfm_app_dist/tasks/utils.py`
- `gridfm_app_dist/tasks/vld_task.py`
- `gridfm_app_dist/training/loss.py`

### Tests

- `tests/conftest.py`
- `tests/test_data_module.py`
- `tests/test_edge_flows.py`
- `tests/test_losses.py`
- `tests/test_node_rank.py`
- `tests/test_pipeline.py`
- `tests/test_simulate_measurements.py`
- `tests/test_yaml_configs.py`

### User documentation

- `README.md`
- `docs/applications/vld_integration.md`

In total, 163 literal occurrences in 38 existing files were replaced. Of these,
159 were standalone package/command identifiers; the other four occurred as a
prefix within longer example or metadata identifiers.

## Verification performed

1. Searched all copied file contents for the previous identifier before this
   report was added: no occurrences remained.
2. Searched all copied path names for the previous identifier: no matching path
   names remained.
3. Confirmed 159 standalone occurrences of `gridfm_app_dist` in the copied code,
   matching the 159 standalone occurrences of the previous identifier in the
   source.
4. Compared 1,169 copied source files with an in-memory expected transformation:
   zero missing files and zero unexpected content mismatches were found.
5. Ran Python bytecode compilation over `gridfm_app_dist/` and `tests/`: passed.
6. Ran setuptools package discovery: it found `gridfm_app_dist` and its five
   subpackages (`datasets`, `io`, `models`, `tasks`, and `training`).

## Test-suite limitation

The complete pytest suite was attempted but could not start because this machine
has Python 3.7.3 and does not have pytest installed. The project declares support
for Python 3.10 through 3.13. No dependencies were installed and no test result
is claimed beyond the syntax-compilation and package-discovery checks above.

## Expected commands after the rename

Examples now use:

```text
gridfm_app_dist --help
python -m gridfm_app_dist --help
```

