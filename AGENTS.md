# AGENTS.md

Compact guidance for OpenCode sessions working in this repo. Read before editing.
Every line here is something an agent would likely get wrong without it.

## What this is

Example scripts and robot assets for the [FIGAROH](https://github.com/thanhndv212/figaroh-plus)
library (robot dynamics identification + geometric calibration). **Not an installable
package** — no `pyproject.toml`/`setup.py`. It is a collection of runnable scripts plus
shared `models/` (robot description packages) and a Viser-based `web-interface/`.

## Setup — conda env is `figaroh-dev`, not what `environment.yml` says

- **All code runs in the `figaroh-dev` conda env.** `conda activate figaroh-dev` first.
  The `environment.yml` in *this* repo names an env `figaroh-examples` — **ignore it**;
  that is not the env anyone uses. The real env is defined in the sibling `figaroh/` repo.
- **`figaroh` is installed editable from the sibling repo** at `../figaroh` (workspace
  layout: `figaroh-ws/{figaroh, figaroh-examples, figaroh-mujoco, ...}`). `import figaroh`
  resolves to `../figaroh/src/figaroh`. After changing core library code, re-run
  `pip install -e .` from `../figaroh` inside `figaroh-dev`.
- `cyipopt` (needed by optimal-trajectory scripts) is conda-only — another reason the env
  is mandatory. See `../figaroh/AGENTS.md` for core-library gotchas.

## Running example scripts — always `cd` into the robot folder first

- Scripts use **relative paths** (`urdf/<file>.urdf`, `config/<file>.yaml`) and
  `package_dirs="../../models"`. They only work when the working directory is
  `examples/<robot>/`:
  ```bash
  conda activate figaroh-dev
  cd examples/ur10
  python calibration.py
  ```
- The correct models path is `"../../models"` — all examples now use this (fixed in
  Phase 1 §1.7). Scripts only work when CWD is `examples/<robot>/`.
- Entry-point scripts: `calibration.py`, `identification.py`, `optimal_config.py`,
  `optimal_trajectory.py`, and `update_model.py` (materializes estimated params into a
  URDF). Not every robot has all of them.
- Robot-specific logic lives in `examples/<robot>/utils/<robot>_tools.py` as subclasses of
  `figaroh.calibration.base_calibration.BaseCalibration`,
  `figaroh.identification.base_identification.BaseIdentification`, and
  `figaroh.optimal.base_optimal_*`.

## Tests & validation — use `validate.py` after every implementation phase

- **Run `python validate.py` from repo root after every phase of implementation work.**
  This is mandatory — it runs both the pytest suite and all example scripts, then
  reports a pass/fail/timeout summary. Do not declare a phase complete until
  `validate.py` passes (or only shows known pre-existing failures).
  ```bash
  conda activate figaroh-dev
  python validate.py                  # tests + all scripts (full)
  python validate.py --quick          # skip slow scripts (optimal_config, optimal_trajectory)
  python validate.py --tests-only     # pytest only
  python validate.py --scripts-only   # example scripts only
  python validate.py --robot ur10     # tests + scripts for one robot
  ```
- `validate.py` sets `MPLBACKEND=Agg` so matplotlib doesn't block on plot windows.
- Slow scripts (optimal_config, optimal_trajectory) use IPOPT and can take 5+ minutes.
  Use `--quick` for fast feedback loops; run full validation before declaring done.
- **Exit code 0 = all pass, 1 = any failure.** Timeouts are reported separately from
  failures — IPOPT timeouts are expected, not bugs.
- `pytest` can also be run directly: `pytest tests/ -v` from repo root.
- `tests/conftest.py` adds `../figaroh/src` **and** the examples root to `sys.path`, so
  tests assume the sibling `figaroh/` repo is checked out next to this one.
- Markers: `slow`, `integration`, `real_config`. Skip slow: `pytest -m 'not slow'`.
- **Do not use `tests/run_tests.py`** — it only runs 3 of the 8 test files and parses
  stdout instead of using the pytest API. It is stale. Use `validate.py` instead.

## Config — two formats coexist; scripts use the unified one

- **Unified** (`*_unified_config.yaml`): `tasks.<task>.*` layout with template inheritance
  via `extends:` (e.g. `extends: "../../templates/manipulator_robot.yaml"`). This is what
  entry-point scripts load.
- **Legacy** (`*_config.yaml`): flat format, auto-detected by `figaroh`'s
  `UnifiedConfigParser`. Keep for backward-compat tests; don't write new configs in it.
- Templates live in `examples/templates/` (`base_robot_config.yaml`,
  `manipulator_robot.yaml`, `humanoid_robot.yaml`). Use `extends:` (not `inherit_from`).
  Keep robot specifics under `robot.properties.*`, task specifics under `tasks.<task>.*`.
- TIAGo has ~10 config files in mixed formats with `extends:` commented out and several
  referencing missing CSVs — treat `examples/tiago/config/` as partially broken.

## Gotchas

- **`examples/shared/` does not exist and is gitignored** (`.gitignore`: `shared/`,
  `examples/shared/`). `examples/__init__.py` still does `from .shared import (...)`
  inside a `try/except`, so it silently no-ops. **Do not import or rely on
  `examples.shared.*`.** (Staubli's dead `ConfigManager` import was removed in Phase 1.)
- **Logging is set to `CRITICAL`** in every entry-point script (`logging.basicConfig(
  level=logging.CRITICAL)`), so debug/info output is suppressed. Change to `INFO` locally
  if you need output; don't assume silence means success. (Phase 3 §3.2 will fix this.)
- **`create_example.sh` has known bugs**: title-casing produces `Ur10` for input `UR10`.
  The "colission" filename typo was fixed in Phase 2. Run it from `examples/`. It
  scaffolds from the TIAGo layout.
- **`.github/` and `.waylog/` are gitignored** — local-only, not shipped. The workflow
  skill at `.github/skills/figaroh-examples-workflow/SKILL.md` is useful but won't exist
  on a fresh clone.
- **No CI, no pre-commit, no lint config** in this repo. The only quality gate is running
  `validate.py` (or `pytest`) manually. The core `figaroh/` repo has the pre-commit hooks,
  not this one. (Phase 6 §6.1 will add CI.)

## Web interface

- Viser-based, under active development (expect breaking changes).
- Run from `web-interface/`: `python main.py` (default `http://localhost:8080`).
  Flags: `--port`, `--host`, `--examples-path`, `--models-path`, `--debug`, `--classic`.
- Known gap: `core/example_loader.py` looks for `config.yaml`/`calibration.yaml` but
  examples use `*_unified_config.yaml`, so config discovery finds nothing until fixed
  (`IMPROVEMENT_PLAN.md` §6.3).

## Branches

- `main` is the default/release branch (currently checked out). `devel` is the dev branch
  — matches the core `figaroh/` repo convention. Other feature branches (`h1v2`, `soarm`,
  `0a`) exist locally.

## Known issues — read `IMPROVEMENT_PLAN.md`

`IMPROVEMENT_PLAN.md` is a 39-item audit (P0–P6) of this repo's current state.
**Phases 1, 2, and 5 are complete** (16 items fixed: failing tests, path bugs,
copy-paste typos, dead code, disconnected workflows). Remaining work:
- **Phase 3** (P2): standardize practices — `if __name__` guards, logging levels,
  argparse/CLI, error handling, move hardcoded params to YAML, type hints
- **Phase 4** (P3): config & model cleanup — consolidate TIAGo configs, deduplicate
  templates, symlink URDFs, standardize paths
- **Phase 6** (P5): infrastructure — CI workflows, fix `run_tests.py`, web-interface
  discovery, templates README, `create_example.sh` bugs
- **Phase 7** (P6): complete incomplete examples — add missing scripts to TALOS and
  Staubli TX40

Consult `IMPROVEMENT_PLAN.md` before assuming a script that errors is your bug — it
may already be a tracked known issue. Run `python validate.py` to check current state.

## CodeGraph

A `.codegraph/` index exists at the repo root. Prefer `codegraph_explore` / `codegraph_node`
(see `~/.config/opencode/AGENTS.md`) over grep/Read for locating symbols and understanding
how example utils wire into `figaroh` base classes.
