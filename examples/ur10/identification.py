import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.ur10.utils.ur10_tools import UR10Identification
from examples.ur10.calibration import (
    export_xml,
    export_urdf_dynamics,
    XML_PATH,
)
from figaroh.tools.robot import load_robot
from figaroh.identification.parameter import get_standard_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UR10 动力学参数辨识(完成后自动更新 URDF + XML)"
    )
    parser.add_argument(
        "--config", type=str,
        default="config/ur10_unified_config.yaml",
        help="统一配置 YAML 路径",
    )
    parser.add_argument(
        "--urdf", type=str,
        default="../../models/ur_description/urdf/ur10_robot.urdf",
        help="原始 URDF 路径",
    )
    parser.add_argument(
        "--xml", type=str,
        default=XML_PATH,
        help="原始 MJCF XML 路径",
    )
    parser.add_argument(
        "--no-update", action="store_true",
        help="只辨识,不自动导出 URDF/XML",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="打印 INFO 级日志"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    os.chdir(Path(__file__).resolve().parent)

    ur10 = load_robot(
        args.urdf,
        package_dirs="../../models",
        load_by_urdf=True,
    )

    ur10_identif = UR10Identification(
        robot=ur10,
        config_file=args.config,
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    active_joints = cfg["robot"]["properties"]["joints"]["active_joints"]

    ps = ur10_identif.identif_config
    ps["active_joints"] = active_joints
    ps["act_Jid"] = [ur10_identif.model.getJointId(i) for i in ps["active_joints"]]
    ps["act_J"] = [ur10_identif.model.joints[jid] for jid in ps["act_Jid"]]
    ps["act_idxq"] = [J.idx_q for J in ps["act_J"]]
    ps["act_idxv"] = [J.idx_v for J in ps["act_J"]]

    if not args.no_update:
        ps["reconstruction"] = {
            "enabled": True,
            "method": "nullspace",
            "prior": {"source": "dict"},
        }
        ps["physical_consistency"] = {
            "enabled": True,
            "mass_min": 0.01,
            "psd_eig_tol": -1e-10,
            "solver": "cvxopt",
            "skip_if_feasible": False,
        }

    ur10_identif.initialize()

    ur10_identif.solve(
        decimate=False,
        plotting=True,
        save_results=False,
    )

    print("\n" + "=" * 60)
    print("UR10 DYNAMIC PARAMETER IDENTIFICATION RESULTS")
    print("=" * 60)
    print(f"Number of base parameters identified: {len(ur10_identif.params_base)}")
    print(f"Correlation coefficient: {ur10_identif.correlation:.4f}")

    print("\nBase parameters:")
    for i, param_name in enumerate(ur10_identif.params_base):
        print(f"{i + 1:2d}. {param_name}: {ur10_identif.phi_base[i]:10.6f}")

    if args.no_update:
        return

    print("\n" + "=" * 60)
    print("Auto-updating model (URDF + XML)...")
    print("=" * 60)

    result = ur10_identif.result
    export_params = get_standard_parameters(
        ur10_identif.model, ur10_identif.identif_config
    )
    export_params.update(result["reconstruction"]["theta_r_dict"])

    pc = result["physical consistency"]
    export_params.update(pc["projected_parameters"])
    print(f"Physical consistency: status='{pc['status']}', "
          f"{len(export_params)} parameters to export.")

    modified_urdf = export_urdf_dynamics(
        args.urdf, export_params, output_path=args.urdf
    )
    modified_xml = export_xml(args.xml, export_params, output_path=args.xml)

    print("\n" + "=" * 60)
    print("Model update complete.")
    print(f"  URDF: {modified_urdf}")
    print(f"  XML:  {modified_xml}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(args)