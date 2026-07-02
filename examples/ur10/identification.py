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

"""
UR10 动力学参数辨识 —— 辨识完成后自动把结果写回 URDF 和 MJCF(XML)。

流程:: 标定 → 辨识 → 重构完整标准参数 → 导出 URDF + XML

    python identification.py                    # 默认:辨识 + 自动更新模型
    python identification.py --no-update        # 只辨识,不更新模型
    python identification.py --verbose          # 详细日志
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path for imports (prefer `pip install -e .` instead)
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.ur10.utils.ur10_tools import UR10Identification  # noqa: E402
from examples.ur10.calibration import export_xml, XML_PATH  # noqa: E402
from figaroh.tools.robot import load_robot  # noqa: E402
from figaroh.tools.urdf_exporter import export_urdf  # noqa: E402
from figaroh.identification.parameter import get_standard_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="UR10 动力学参数辨识(完成后自动更新 URDF + XML)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/ur10_unified_config.yaml",
        help="统一配置 YAML 路径",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="urdf/ur10_robot.urdf",
        help="原始 URDF 路径",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=XML_PATH,
        help="原始 MJCF XML 路径",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="只辨识,不自动导出 URDF/XML",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="打印 INFO 级日志"
    )
    return parser.parse_args()


def _timestamp() -> str:
    """时间戳字符串(用于文件名)。"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _export_models(
    params: dict,
    urdf_path: str,
    xml_path: str,
    *,
    verbose: bool = False,
) -> tuple[str | None, str | None]:
    """把辨识参数写回 URDF 和 MJCF XML。

    Args:
        params: {参数名: 值}。
        urdf_path: 原始 URDF 路径。
        xml_path: 原始 XML 路径。
        verbose: 详细日志。

    Returns:
        (modified_urdf, modified_xml),失败则对应项为 None。
    """
    ts = _timestamp()
    urdf_out = str(Path(urdf_path).parent / f"ur10_robot_modified_{ts}.urdf")
    xml_out = str(Path(xml_path).parent / f"ur10e_modified_{ts}.xml")

    modified_urdf = None
    modified_xml = None
    try:
        modified_urdf = export_urdf(
            urdf_path, params, output_path=urdf_out, verbose=verbose
        )
        print(f"URDF exported: {modified_urdf}")
    except Exception as e:
        print(f"Warning: URDF export failed: {e}", file=sys.stderr)

    try:
        if Path(xml_path).exists():
            modified_xml = export_xml(
                xml_path, params, output_path=xml_out, verbose=verbose
            )
    except Exception as e:
        print(f"Warning: XML export failed: {e}", file=sys.stderr)

    return modified_urdf, modified_xml


def main(args: argparse.Namespace) -> None:
    """UR10 动力学辨识主函数。"""
    # 切换到脚本所在目录,保证相对路径能找到
    os.chdir(Path(__file__).resolve().parent)

    # 校验输入文件
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # 加载 UR10 模型
        ur10 = load_robot(
            args.urdf,
            package_dirs="../../models",
            load_by_urdf=True,
        )

        # 创建辨识对象
        ur10_identif = UR10Identification(
            robot=ur10,
            config_file=args.config,
        )

        # 从统一配置读 active_joints
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        active_joints = cfg["robot"]["properties"]["joints"]["active_joints"]

        ps = ur10_identif.identif_config
        ps["active_joints"] = active_joints
        # 关节参数
        ps["act_Jid"] = [ur10_identif.model.getJointId(i) for i in ps["active_joints"]]
        ps["act_J"] = [ur10_identif.model.joints[jid] for jid in ps["act_Jid"]]
        ps["act_idxq"] = [J.idx_q for J in ps["act_J"]]
        ps["act_idxv"] = [J.idx_v for J in ps["act_J"]]

        # 默认开启重构,把基参数还原成完整标准参数(写回 URDF/XML 必需)
        if not args.no_update:
            if not ps.get("reconstruction"):
                ps["reconstruction"] = {"enabled": True, "method": "nullspace"}

        # 初始化(含数据处理)
        ur10_identif.initialize()

        # 辨识
        ur10_identif.solve(
            decimate=False,
            plotting=True,
            save_results=False,
        )

        # 打印结果摘要
        print("\n" + "=" * 60)
        print("UR10 DYNAMIC PARAMETER IDENTIFICATION RESULTS")
        print("=" * 60)

        print(
            f"Number of base parameters identified: " f"{len(ur10_identif.params_base)}"
        )
        print(f"Correlation coefficient: {ur10_identif.correlation:.4f}")

        if hasattr(ur10_identif, "result"):
            for key, value in ur10_identif.result.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        print(f"{key}: {value:.6f}")
                    else:
                        print(f"{key}: {value}")
                else:
                    print(f"{key}: {type(value).__name__} of length {len(value)}")

        print("\nBase parameters:")
        for i, param_name in enumerate(ur10_identif.params_base):
            print(f"{i + 1:2d}. {param_name}: {ur10_identif.phi_base[i]:10.6f}")

        print("\nIdentification completed successfully!")

        # ── 自动更新模型:导出 URDF + XML ──
        if not args.no_update:
            print("\n" + "=" * 60)
            print("Auto-updating model (URDF + XML)...")
            print("=" * 60)

            recon = (ur10_identif.result or {}).get("reconstruction", {})
            recon_status = recon.get("status", "unavailable")
            theta_r_dict = recon.get("theta_r_dict", {})

            if recon_status != "ok" or not theta_r_dict:
                print(
                    f"Warning: reconstruction status='{recon_status}', "
                    "cannot export full standard parameters."
                )
                print("  Enable reconstruction in config or check data quality.")
                return

            # 合并:重构出来的参数覆盖标称值,缺失的用标称标准参数补全
            std_params = get_standard_parameters(
                ur10_identif.model, ur10_identif.identif_config
            )
            export_params = dict(std_params)
            export_params.update(theta_r_dict)

            print(f"Reconstructed {len(theta_r_dict)} standard parameters.")
            print(f"Total export parameters (with nominal fallback): {len(export_params)}")

            modified_urdf, modified_xml = _export_models(
                export_params,
                args.urdf,
                args.xml,
                verbose=args.verbose,
            )

            print("\n" + "=" * 60)
            print("Model update complete.")
            if modified_urdf:
                print(f"  Nominal URDF:   {args.urdf}")
                print(f"  Modified URDF:  {modified_urdf}")
            if modified_xml:
                print(f"  Nominal XML:    {args.xml}")
                print(f"  Modified XML:   {modified_xml}")
            print("=" * 60)

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