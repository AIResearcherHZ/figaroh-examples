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

import logging
import sys
from pathlib import Path

# Configure logging at application entry point
# Levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
logging.basicConfig(
    level=logging.CRITICAL,  # Suppress logging output
    format="%(name)s - %(levelname)s - %(message)s",
)

# Add project root to path for imports (prefer `pip install -e .` instead)
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.ur10.utils.ur10_tools import (
    OptimalTrajectoryIPOPT
)
from figaroh.tools.robot import load_robot


def main():
    """Main function for UR10 optimal trajectory generation."""

    # Load UR10 robot model
    ur10 = load_robot(
        "urdf/ur10_robot.urdf",
        package_dirs="../../models",
        load_by_urdf=True,
    )
    active_joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    # Create optimal trajectory object
    ur10_traj = OptimalTrajectoryIPOPT(
        robot=ur10, active_joints=active_joints,
        config_file="config/ur10_unified_config.yaml"
    )
    ps = ur10_traj.identif_config

    # Joint parameters
    ps["active_joints"] = active_joints
    ps["act_Jid"] = [
        ur10_traj.model.getJointId(i) for i in ps["active_joints"]
    ]
    ps["act_J"] = [ur10_traj.model.joints[jid] for jid in ps["act_Jid"]]
    ps["act_idxq"] = [J.idx_q for J in ps["act_J"]]
    ps["act_idxv"] = [J.idx_v for J in ps["act_J"]]

    # Initialize
    ur10_traj.initialize()

    # Generate optimal trajectory
    optimal_trajectory = ur10_traj.solve(stack_reps=2)

    if optimal_trajectory is not None:
        # Display results
        print("Optimal trajectory generation completed successfully!")
        # Plot and save results
        ur10_traj.plot_results()
    else:
        print("Failed to generate optimal trajectory. Check constraints and parameters.")


if __name__ == "__main__":
    main()
