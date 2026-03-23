# TALOS example (torso/arm calibration)

This folder contains a runnable example for **kinematic calibration** of the TALOS humanoid robot’s torso/arm kinematic chain using FIGAROH.

## What’s included

- `calibration_upperbody.py`: entry point for the TALOS torso-arm calibration workflow.
- `utils/talos_tools.py`: `TALOSCalibration`, a TALOS-specific specialization of `figaroh.calibration.base_calibration.BaseCalibration`.
- `config/`: YAML configuration files (typically use `talos_unified_config.yaml`).
- `data/`: example measurement CSV(s) used by the calibration.
- `urdf/`: the TALOS URDF(s) used by the scripts.
- `update_model.py`: helper for writing estimated offsets to `data/offset.xacro` and `data/offset.yaml`.

## Run

Most scripts assume the working directory is this folder:

```bash
cd examples/talos
python calibration_upperbody.py
```

The script loads `urdf/talos_full_v2.urdf` and uses `models/` (at the repo root) as the URDF package directory.

## Outputs

- Calibration runs the optimizer and prints the list of calibrated parameter names.
- If you want to materialize the estimated offsets into a file (for downstream URDF/xacro use), `update_model.py` writes:
  - `data/offset.xacro`
  - `data/offset.yaml`

## Notes

- `calibration_upperbody.py` sets `known_baseframe` and `known_tipframe` flags explicitly before initialization.
- The calibration config and data format are governed by the YAML under `config/`. If you change the dataset, update the YAML paths/fields accordingly.
