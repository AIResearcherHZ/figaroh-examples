# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import logging
import sys
import yaml
from pathlib import Path

project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.ur10.utils.ur10_tools import OptimalTrajectoryIPOPT
from figaroh.tools.robot import load_robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR10 optimal trajectory generation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ur10_unified_config.yaml",
        help="Path to unified config YAML file",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="../../models/ur_description/urdf/ur10_robot.urdf",
        help="Path to robot URDF file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (INFO) logging"
    )
    return parser.parse_args()


def save_trajectory_csv(
    results: dict, out_dir: str = "results", name: str = "ur10_optimal_trajectory"
) -> str:
    import os

    import numpy as np
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for si, (T, P, V, A) in enumerate(
        zip(results["T_F"], results["P_F"], results["V_F"], results["A_F"])
    ):
        d = {"segment": si, "time": np.asarray(T).reshape(-1)}
        for j in range(P.shape[1]):
            d[f"q{j + 1}"] = P[:, j]
        for j in range(V.shape[1]):
            d[f"dq{j + 1}"] = V[:, j]
            d[f"ddq{j + 1}"] = A[:, j]
        frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"轨迹已保存: {csv_path}  ({len(df)} 采样点)")
    return csv_path


def main(args: argparse.Namespace) -> None:
    import os
    os.chdir(Path(__file__).resolve().parent)
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        ur10 = load_robot(
            args.urdf,
            package_dirs="../../models",
            load_by_urdf=True,
        )

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        active_joints = cfg["robot"]["properties"]["joints"]["active_joints"]

        ur10_traj = OptimalTrajectoryIPOPT(
            robot=ur10,
            active_joints=active_joints,
            config_file=args.config,
        )
        ps = ur10_traj.identif_config

        ps["active_joints"] = active_joints
        ps["act_Jid"] = [ur10_traj.model.getJointId(i) for i in ps["active_joints"]]
        ps["act_J"] = [ur10_traj.model.joints[jid] for jid in ps["act_Jid"]]
        ps["act_idxq"] = [J.idx_q for J in ps["act_J"]]
        ps["act_idxv"] = [J.idx_v for J in ps["act_J"]]

        ur10_traj.initialize()

        optimal_trajectory = ur10_traj.solve(stack_reps=2)

        if optimal_trajectory is not None and optimal_trajectory.get("T_F"):
            print("Optimal trajectory generation completed successfully!")
            save_trajectory_csv(optimal_trajectory)
            ur10_traj.plot_results()
        else:
            print(
                "Failed to generate optimal trajectory. Check constraints and parameters."
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(args)
