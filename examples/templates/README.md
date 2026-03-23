# Example templates

This folder contains YAML templates used as starting points when creating new robot examples and when structuring unified configuration files.

## Files

- `base_robot_config.yaml`: common fields shared across robot types.
- `manipulator_robot.yaml`: additional fields typically relevant for fixed-base manipulators.
- `humanoid_robot.yaml`: additional fields typically relevant for humanoids (multiple chains, markers, etc.).

## How this is used

- `../create_example.sh` scaffolds a new example folder (based on the TIAGo example layout). These templates are provided as reference for how to organize robot configuration files.
- You can also copy/paste sections into a robot’s `config/*_unified_config.yaml` and then customize paths and robot-specific settings.
