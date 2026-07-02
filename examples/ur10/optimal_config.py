import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.ur10.utils.ur10_tools import UR10OptimalCalibration
from figaroh.tools.robot import load_robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UR10 optimal configuration generation"
    )
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
        print("UR10 Optimal Configuration Generation")
        print("=" * 50)

        print("\n1. Loading UR10 robot model...")
        robot = load_robot(
            args.urdf,
            package_dirs="../../models",
            load_by_urdf=True,
        )

        print("\n2. Setting up optimal calibration...")
        opt_calib = UR10OptimalCalibration(robot, args.config)

        print(f"Calibration model: {opt_calib.calib_config['calib_model']}")
        print(f"Minimum configurations required: {opt_calib.minNbChosen}")

        print("\n3. Solving optimal configuration selection...")

        start_time = time.time()
        opt_calib.solve(save_file=True)
        solve_time = time.time() - start_time

        print(f"Optimization completed in {solve_time:.2f} seconds")

        print("\n4. Results Summary:")

        if (
            hasattr(opt_calib, "optimal_configurations")
            and "calibration_joint_configurations" in opt_calib.optimal_configurations
        ):
            selected_configs = opt_calib.optimal_configurations[
                "calibration_joint_configurations"
            ]
            print(f"Selected {len(selected_configs)} optimal configurations")
            print(f"Total candidates: {opt_calib.calib_config['NbSample']}")
            ratio = len(selected_configs) / opt_calib.calib_config["NbSample"]
            print(f"Selection ratio: {ratio:.2%}")

        if hasattr(opt_calib, "detroot_whole"):
            print(f"Information matrix determinant root: {opt_calib.detroot_whole:.4e}")

        if hasattr(opt_calib, "optimal_weights"):
            weights = opt_calib.optimal_weights
            if hasattr(weights, "__len__") and len(weights) > 0:
                print(f"Weight sum: {np.sum(weights):.4f}")

        print("\n" + "=" * 50)
        print("UR10 Optimal Configuration Generation Completed!")
        print("Results saved to 'results/' directory")
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
