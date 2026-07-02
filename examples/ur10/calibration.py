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
UR10 calibration, URDF update, and validation — all-in-one entry-point.

Modes::

    # Full pipeline: calibrate → plot → save → export → viser viz
    python calibration.py

    # Calibrate only (save results with timestamp, skip export)
    python calibration.py --calibrate-only

    # Load saved results → export URDF → verify FK
    python calibration.py --update-model

    # Visually validate a previously exported modified URDF
    python calibration.py --viz-validation
    python calibration.py --viz-validation --model path/to/modified.urdf

    # Interactive step selection
    python calibration.py --interactive

Run ``python calibration.py --help`` for all available flags.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path for imports (prefer `pip install -e .` instead)
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.ur10.utils.ur10_tools import UR10Calibration  # noqa: E402
from figaroh.tools.robot import load_robot  # noqa: E402
from figaroh.tools.urdf_exporter import (  # noqa: E402
    export_urdf,
    frame_settings_doc,
)
from figaroh.tools.export_validation import URDFComparison  # noqa: E402

logger = logging.getLogger(__name__)

DATA_DIR = "data/calibration"
URDF_STEM = "ur10_robot"  # stem for discovering modified URDFs
# 默认 MJCF 模型(与 replay_mujoco.py 一致)
XML_PATH = "../../models/ur_description/ur10e.xml"


# ── MJCF(XML)导出 ──────────────────────────────────────────────────


def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """RPY(弧度)转四元数 [w, x, y, z](MuJoCo 顺序)。"""
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 [w,x,y,z]。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _parse_float(val) -> float:
    """安全转 float。"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def export_xml(
    nominal_xml_path: str,
    params: dict,
    *,
    output_path: str | None = None,
    verbose: bool = False,
) -> str:
    """把标定/辨识参数写回 MuJoCo MJCF(XML)。

    与 export_urdf 对应,支持以下参数:
      - 几何偏移(叠加): d_px/d_py/d_pz/d_phix/d_phiy/d_phiz_{joint}
      - 质量(绝对): m_{body}
      - 一阶矩(绝对): mx/my/mz_{body} → COM = (mx/m, my/m, mz/m)
      - 惯量(绝对): Ixx/Ixy/Ixz/Iyy/Iyz/Izz_{body}
        非对角为零用 diaginertia,否则用 fullinertia
      - 摩擦(绝对): fv_{joint} → damping, fs_{joint} → frictionloss
      - 转子惯量(绝对): Ia_{joint} → armature
      - 测量坐标系参数(base_*, pEE*, phiEE*)不写入 MJCF

    Args:
        nominal_xml_path: 原始 MJCF 路径。
        params: {参数名: 值}。
        output_path: 输出路径,None 则用 <stem>_modified_<时间戳>.xml。
        verbose: 打印应用了哪些参数。

    Returns:
        输出 XML 的绝对路径。
    """
    import xml.etree.ElementTree as ET

    nominal = Path(nominal_xml_path)
    if not nominal.exists():
        raise FileNotFoundError(f"MJCF not found: {nominal}")

    if output_path is None:
        ts = _timestamp_str()
        output_path = str(nominal.parent / f"{nominal.stem}_modified_{ts}.xml")
    out = Path(output_path)

    tree = ET.parse(str(nominal))
    root = tree.getroot()

    # 建立 body name → element 的索引
    body_map: dict[str, ET.Element] = {}
    for body in root.iter("body"):
        name = body.get("name")
        if name:
            body_map[name] = body

    # worldbody 下的第一个 body(base)也可能需要处理
    worldbody = root.find("worldbody")

    def _get_body(name: str) -> ET.Element | None:
        return body_map.get(name)

    def _get_or_create_inertial(body: ET.Element) -> ET.Element:
        ine = body.find("inertial")
        if ine is None:
            ine = ET.SubElement(body, "inertial")
        return ine

    def _get_joint(body: ET.Element, name: str) -> ET.Element | None:
        for jt in body.findall("joint"):
            if jt.get("name") == name:
                return jt
        return None

    # ── 按类别收集参数 ──
    # 几何偏移(叠加到 body 的 pos/quat)
    placement: dict[str, list[float]] = {}  # joint → [dx,dy,dz,dphix,dphiy,dphiz]
    for name, val in params.items():
        for axis, idx in [("d_px", 0), ("d_py", 1), ("d_pz", 2),
                          ("d_phix", 3), ("d_phiy", 4), ("d_phiz", 5)]:
            if name.startswith(f"{axis}_"):
                target = name[len(axis) + 1:]
                if target:
                    placement.setdefault(target, [0.0] * 6)[idx] = _parse_float(val)
                break

    # 惯性参数(绝对): m_, mx_, my_, mz_, Ixx_... Izz_
    mass: dict[str, float] = {}
    moments: dict[str, list[float]] = {}  # body → [mx, my, mz]
    inertia: dict[str, np.ndarray] = {}   # body → 3x3 惯量矩阵
    for name, val in params.items():
        v = _parse_float(val)
        if name.startswith("m_"):
            mass[name[2:]] = v
        elif name.startswith("mx_"):
            moments.setdefault(name[3:], [0.0, 0.0, 0.0])[0] = v
        elif name.startswith("my_"):
            moments.setdefault(name[3:], [0.0, 0.0, 0.0])[1] = v
        elif name.startswith("mz_"):
            moments.setdefault(name[3:], [0.0, 0.0, 0.0])[2] = v
        elif name.startswith("Ixx_"):
            inertia.setdefault(name[4:], np.zeros((3, 3)))[0, 0] = v
        elif name.startswith("Iyy_"):
            inertia.setdefault(name[4:], np.zeros((3, 3)))[1, 1] = v
        elif name.startswith("Izz_"):
            inertia.setdefault(name[4:], np.zeros((3, 3)))[2, 2] = v
        elif name.startswith("Ixy_"):
            inertia.setdefault(name[4:], np.zeros((3, 3)))[0, 1] = v
            inertia[name[4:]][1, 0] = v
        elif name.startswith("Ixz_"):
            inertia.setdefault(name[4:], np.zeros((3, 3)))[0, 2] = v
            inertia[name[4:]][2, 0] = v
        elif name.startswith("Iyz_"):
            inertia.setdefault(name[4:], np.zeros((3, 3)))[1, 2] = v
            inertia[name[4:]][2, 1] = v

    # 摩擦/转子惯量(绝对): fv_, fs_, Ia_
    friction: dict[str, dict] = {}  # joint → {damping, frictionloss, armature}
    for name, val in params.items():
        v = _parse_float(val)
        if name.startswith("fv_"):
            friction.setdefault(name[3:], {})["damping"] = v
        elif name.startswith("fs_"):
            friction.setdefault(name[3:], {})["frictionloss"] = v
        elif name.startswith("Ia_"):
            friction.setdefault(name[3:], {})["armature"] = v

    applied = 0

    # ── 应用几何偏移 ──
    for target, deltas in placement.items():
        body = _get_body(target)
        if body is None:
            if verbose:
                logger.warning("XML body '%s' not found, skipping", target)
            continue
        # pos 叠加
        cur_pos = [float(x) for x in body.get("pos", "0 0 0").split()]
        while len(cur_pos) < 3:
            cur_pos.append(0.0)
        new_pos = [cur_pos[i] + deltas[i] for i in range(3)]
        body.set("pos", " ".join(_fmt_xml(x) for x in new_pos))
        # quat 叠加(RPY 增量转四元数后左乘)
        cur_quat = [float(x) for x in body.get("quat", "1 0 0 0").split()]
        while len(cur_quat) < 4:
            cur_quat.extend([0.0] * (4 - len(cur_quat)))
        cur_q = np.array(cur_quat)
        d_rpy = np.array(deltas[3:6])
        if np.any(d_rpy != 0):
            dq = _rpy_to_quat(d_rpy)
            new_q = _quat_mul(dq, cur_q)
            new_q = new_q / np.linalg.norm(new_q)
            body.set("quat", " ".join(_fmt_xml(x) for x in new_q))
        applied += 1
        if verbose:
            logger.info("XML body '%s' pos/quat updated", target)

    # ── 应用惯性参数 ──
    all_bodies = set(mass) | set(moments) | set(inertia)
    for bname in all_bodies:
        body = _get_body(bname)
        if body is None:
            if verbose:
                logger.warning("XML body '%s' not found, skipping", bname)
            continue
        ine = _get_or_create_inertial(body)
        m = mass.get(bname)
        if m is not None:
            ine.set("mass", _fmt_xml(m))
        # COM = 一阶矩 / 质量
        if bname in moments and m and m != 0:
            mx, my, mz = moments[bname]
            com = [mx / m, my / m, mz / m]
            ine.set("pos", " ".join(_fmt_xml(x) for x in com))
        # 惯量
        if bname in inertia:
            I = inertia[bname]
            Ixx, Iyy, Izz = I[0, 0], I[1, 1], I[2, 2]
            Ixy, Ixz, Iyz = I[0, 1], I[0, 2], I[1, 2]
            if abs(Ixy) < 1e-15 and abs(Ixz) < 1e-15 and abs(Iyz) < 1e-15:
                ine.set("diaginertia",
                        " ".join(_fmt_xml(x) for x in [Ixx, Iyy, Izz]))
                ine.attrib.pop("fullinertia", None)
            else:
                # MuJoCo fullinertia: Ixx Iyy Izz Ixy Ixz Iyz
                ine.set("fullinertia",
                        " ".join(_fmt_xml(x)
                                 for x in [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]))
                ine.attrib.pop("diaginertia", None)
        applied += 1
        if verbose:
            logger.info("XML body '%s' inertia updated", bname)

    # ── 应用摩擦/转子惯量 ──
    for jname, attrs in friction.items():
        body = _get_body(jname)
        if body is None:
            if verbose:
                logger.warning("XML body '%s' not found, skipping", jname)
            continue
        jt = _get_joint(body, jname)
        if jt is None:
            # 关节名可能就是 body 名,尝试在 body 内找任意 joint
            joints = body.findall("joint")
            if joints:
                jt = joints[0]
        if jt is None:
            continue
        if "damping" in attrs:
            jt.set("damping", _fmt_xml(attrs["damping"]))
        if "frictionloss" in attrs:
            jt.set("frictionloss", _fmt_xml(attrs["frictionloss"]))
        if "armature" in attrs:
            jt.set("armature", _fmt_xml(attrs["armature"]))
        applied += 1
        if verbose:
            logger.info("XML joint '%s' dynamics updated", jname)

    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), xml_declaration=True, encoding="utf-8")
    print(f"XML exported: {out}  ({applied} params applied)")
    return str(out.resolve())


def _fmt_xml(v: float) -> str:
    """格式化浮点数,紧凑无科学记数法。"""
    if v == 0.0:
        return "0"
    s = f"{v:.10g}"
    if "e" in s or "E" in s:
        s = f"{v:.10f}".rstrip("0").rstrip(".")
    return s


# ── Timestamp and file discovery helpers ────────────────────────────


def _timestamp_str() -> str:
    """Return a compact timestamp string for filenames (YYYYMMDD_HHMMSS)."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _discover_npz_files(data_dir: str = DATA_DIR) -> list[Path]:
    """Return sorted list of calibration .npz files, newest last."""
    files = sorted(Path(data_dir).glob("calibration_results_*.npz"))
    if not files:
        files = sorted(Path("data").glob("calibration_results_*.npz"))
    if not files:
        for p in [Path(data_dir) / "calibration_results.npz",
                  Path("data") / "calibration_results.npz"]:
            if p.exists():
                files = [p]
                break
    return files


def _discover_modified_urdf_files(stem: str = URDF_STEM) -> list[Path]:
    """Return sorted list of modified URDF files, newest last."""
    files = sorted(Path("urdf").glob(f"{stem}_modified_*.urdf"))
    if not files:
        legacy = Path(f"urdf/{stem}_modified.urdf")
        if legacy.exists():
            files = [legacy]
    return files


def _select_npz(data_dir: str = DATA_DIR) -> str:
    """Select a calibration .npz file. Interactive if TTY, else latest."""
    files = _discover_npz_files(data_dir)
    if not files:
        print(
            f"Error: No calibration results found in {data_dir}/ or data/.",
            file=sys.stderr,
        )
        print(
            "  Run `python calibration.py --calibrate-only` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not sys.stdin.isatty():
        return str(files[-1])

    print("\nAvailable calibration results:")
    print(f"  [0]  {files[-1].name}  (latest)")
    for i, f in enumerate(files):
        if f != files[-1]:
            info = ""
            try:
                d = np.load(f)
                info = f"  ({len(d.get('param_names', []))} params)"
            except Exception:
                pass
            print(f"  [{i + 1}]  {f.name}{info}")
    prompt = f"Select result [0-{len(files)}], default=0: "
    choice = input(prompt).strip()
    idx = int(choice) if choice else 0
    return str(files[-1] if idx == 0 else files[idx - 1])


def _select_modified_urdf(stem: str = URDF_STEM) -> str:
    """Select a modified URDF. Interactive if TTY, else latest."""
    files = _discover_modified_urdf_files(stem)
    if not files:
        print(
            f"Error: No modified URDFs found matching 'urdf/{stem}_modified_*'.",
            file=sys.stderr,
        )
        print(
            "  Run `python calibration.py` or `--update-model` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not sys.stdin.isatty():
        return str(files[-1])

    print("\nAvailable modified URDFs:")
    print(f"  [0]  {files[-1].name}  (latest)")
    for i, f in enumerate(files):
        if f != files[-1]:
            print(f"  [{i + 1}]  {f.name}")
    prompt = f"Select model [0-{len(files)}], default=0: "
    choice = input(prompt).strip()
    idx = int(choice) if choice else 0
    return str(files[-1] if idx == 0 else files[idx - 1])


# ── Step selection (--interactive) ──────────────────────────────────


def _select_steps() -> list[str]:
    """Prompt user which steps to include. Returns list of step keys."""
    steps_info = [
        ("calibrate", "Calibration (required if no saved results exist)"),
        ("export", "Export URDF"),
        ("verify", "URDF Consistency"),
        ("viz", "Viser Visualization"),
    ]
    print("\nSelect steps to include (comma-separated numbers, 'all', or 'done'):")
    for i, (key, desc) in enumerate(steps_info, 1):
        print(f"  [{i}] {desc}")
    print()
    choice = input("Steps: ").strip().lower()

    if choice == "all":
        return [s[0] for s in steps_info]
    if not choice:
        return [s[0] for s in steps_info]

    selected = []
    try:
        for token in re.split(r"[,\s]+", choice):
            token = token.strip()
            if not token:
                continue
            idx = int(token) - 1
            if 0 <= idx < len(steps_info):
                selected.append(steps_info[idx][0])
    except ValueError:
        print(f"Invalid input: {choice}. Running all steps.")
        return [s[0] for s in steps_info]
    return selected


# ── Parsing ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "UR10 calibration, URDF update, and validation — all-in-one entry-point. "
            "Default mode runs the full pipeline: calibrate → plot → save → export → "
            "visualize."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  (default)         Full pipeline: calibrate, plot, save, export, viz\n"
            "  --calibrate-only  Calibrate + plot + save (no export, no viz)\n"
            "  --update-model    Load saved results → export URDF → verify FK\n"
            "  --viz-validation  Visually validate a previously exported URDF\n"
            "  --interactive     Select which steps to run interactively\n"
            "\n"
            "All saved files are timestamped to avoid overwriting.\n"
            "Use --model <path> with --viz-validation to skip file selection."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/ur10_unified_config.yaml",
        help="Path to unified config YAML file (default: %(default)s)",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="urdf/ur10_robot.urdf",
        help="Path to robot URDF file (nominal model) (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for modified URDF in export mode. "
            "Default: urdf/<stem>_modified_<timestamp>.urdf"
        ),
    )
    parser.add_argument(
        "--update-model",
        action="store_true",
        help=(
            "Load saved calibration results → export URDF → verify FK. "
            "Prompts to select which .npz file to use (interactive) or "
            "picks the latest (non-TTY)."
        ),
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help=(
            "Run calibration with plotting, save results (timestamped .npz), "
            "then exit with a reminder about --update-model."
        ),
    )
    parser.add_argument(
        "--viz-validation",
        action="store_true",
        help=(
            "Visually validate a previously exported modified URDF via "
            "viser.  Use --model to specify the file path; otherwise an "
            "interactive menu lists available modified URDFs."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to a modified URDF for --viz-validation.  Overrides "
            "interactive file selection."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively select which steps to include (calibrate, export, "
        "verify, viz).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress matplotlib plots during calibration (useful in CI).",
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        default=None,
        help=(
            "Path to a separate validation measurement CSV. "
            "Overrides validation_data_file in config."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (INFO) logging.",
    )
    return parser.parse_args()


# ── Calibration ─────────────────────────────────────────────────────


def _run_calibration(
    urdf_path: str,
    config_path: str,
    *,
    plot: bool = True,
    verbose: bool = False,
    validation_data: str | None = None,
) -> tuple[np.ndarray, list[str], str]:
    """Run UR10 calibration.

    Returns (result_vector, param_names, saved_path) where *saved_path*
    is the timestamped .npz path.
    """
    ur10 = load_robot(urdf_path, package_dirs="../../models", load_by_urdf=True)
    ur10_calib = UR10Calibration(ur10, config_path)
    ur10_calib.calib_config["known_baseframe"] = False
    ur10_calib.calib_config["known_tipframe"] = False
    if validation_data:
        ur10_calib.calib_config["validation_data_file"] = validation_data
    ur10_calib.initialize()
    result = ur10_calib.solve(plotting=plot, enable_logging=verbose)
    param_names = ur10_calib.calib_config["param_name"]

    # Save with timestamp
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = _timestamp_str()
    saved_path = os.path.join(DATA_DIR, f"calibration_results_{ts}.npz")
    np.savez(saved_path, result=result.x, param_names=param_names)
    print(f"Calibration results saved to {saved_path}")

    # Print log-map residual statistics (full 6-DOF: position + orientation)
    PEE_est = ur10_calib.get_pose_from_measure(result.x)
    residuals = ur10_calib._compute_logmap_residuals(
        ur10_calib.PEE_measured, PEE_est
    )
    calib_config = ur10_calib.calib_config
    n_dofs = calib_config["calibration_index"]
    n_samples = calib_config["NbSample"]
    residuals_2d = residuals.reshape((n_dofs, n_samples))
    rmse_pos = np.sqrt(np.mean(np.sum(residuals_2d[:3] ** 2, axis=0)))
    rmse_orient = np.sqrt(np.mean(np.sum(residuals_2d[3:] ** 2, axis=0)))
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"\nPost-calibration residual statistics (log map):")
    print(f"  Position RMSE:    {rmse_pos * 1000:.2f} mm")
    print(f"  Orientation RMSE: {rmse_orient * 180 / np.pi:.4f} deg")
    print(f"  Overall RMSE:     {rmse:.6f}")

    return result.x, param_names, saved_path


# ── Export + verify ─────────────────────────────────────────────────


def export_with_verification(
    params: dict,
    nominal_urdf: str | Path,
    *,
    output_path: str | Path | None = None,
    nominal_xml: str | Path | None = None,
    xml_output_path: str | Path | None = None,
    calibration_type: str = "mocap",
    verbose: bool = False,
) -> tuple[str, str | None, URDFComparison, object]:
    """Export URDF(和可选的 MJCF XML)并验证 FK。

    Args:
        params: {参数名: 值}(关节级 + 坐标系参数)。
                坐标系参数自动识别,不写入模型。
        nominal_urdf: 原始 URDF 路径。
        output_path: 修改后 URDF 的输出路径。
                     None 则自动生成 urdf/<stem>_modified_<ts>.urdf。
        nominal_xml: 原始 MJCF XML 路径;None 则不导出 XML。
        xml_output_path: 修改后 XML 的输出路径;None 则自动生成。
        calibration_type: 传给 frame_settings_doc()。
        verbose: 打印详细日志。

    Returns:
        (modified_urdf_path, modified_xml_path_or_None, comparison, errors)
    """
    nominal_path = Path(nominal_urdf)

    # 生成带时间戳的 URDF 输出路径
    if output_path is None:
        stem = nominal_path.stem
        ts = _timestamp_str()
        output_path = str(nominal_path.parent / f"{stem}_modified_{ts}.urdf")

    # 导出 URDF — export_urdf() 自动区分关节参数和坐标系参数
    modified_path = export_urdf(
        str(nominal_path),
        params,
        output_path=str(output_path),
        verbose=verbose,
    )

    # 同步导出 MJCF XML(如果指定了原始 XML)
    modified_xml = None
    if nominal_xml and Path(nominal_xml).exists():
        modified_xml = export_xml(
            str(nominal_xml),
            params,
            output_path=str(xml_output_path) if xml_output_path else None,
            verbose=verbose,
        )

    # 显示测量坐标系文档
    frame_settings_doc(calibration_type=calibration_type, verbose=verbose)

    # 打印测量坐标系参数(不自动写入)
    frame_params = {
        k: v
        for k, v in params.items()
        if k.startswith(("base_", "pEE", "phiEE"))
    }
    if frame_params:
        print("\nMetrology frame parameters (NOT auto-applied to URDF):")
        for k, v in sorted(frame_params.items()):
            print(f"  {k} = {v:.6f}")
        print(
            "  → These define the calibration-setup transform and must be\n"
            "    configured in your controller or pipeline. See\n"
            "    frame_settings_doc() for defaults and explanations."
        )
    else:
        print("\nNo metrology frame parameters in calibration result.")

    # URDF 导出一致性检查

    print("URDF export consistency check (nominal vs. exported URDF)")
    print("=" * 60)
    comp = URDFComparison(str(nominal_path), modified_path)
    errors = comp.fk_consistency_check(n_samples=200)
    print(f"  Position RMSE:    {errors.rmse_position * 1000:.2f} mm")
    print(f"  Orientation RMSE: {errors.rmse_orientation * 180 / np.pi:.4f} deg")
    print(f"  Max position:     {errors.max_position * 1000:.2f} mm")
    print(f"  Max orientation:  {errors.max_orientation * 180 / np.pi:.4f} deg")
    print(f"  (samples: {len(errors.per_sample)})")

    return modified_path, modified_xml, comp, errors


# ── Visual validation ───────────────────────────────────────────────


def show_validation(comp: URDFComparison):
    """Open interactive viser validation with trajectory, static comparison,
    error plots, replay, and opacity controls.

    See :meth:`URDFComparison.show_interactive_validation` for details.
    """
    try:
        import viser  # noqa: F401
    except ImportError:
        print("  viser not installed; skipping visual validation.")
        return

    comp.show_interactive_validation(n_trajectory=50, port=8080)


# ── Mode handlers ───────────────────────────────────────────────────


def _run_update_model(args: argparse.Namespace) -> None:
    """加载 .npz → 导出 URDF + XML → 验证 FK。"""
    npz_path = args.model if args.model else _select_npz()
    print(f"\nLoading calibration results from: {npz_path}")
    data = np.load(npz_path)
    result_x = data["result"]
    param_names = list(data["param_names"])
    params = dict(zip(param_names, result_x))
    print(f"Loaded {len(params)} calibration parameters.")

    modified_urdf, modified_xml, comp, errors = export_with_verification(
        params,
        str(args.urdf),
        output_path=args.output,
        nominal_xml=XML_PATH,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("Update complete.")
    print(f"  Source results: {npz_path}")
    print(f"  Nominal URDF:   {args.urdf}")
    print(f"  Modified URDF:  {modified_urdf}")
    if modified_xml:
        print(f"  Nominal XML:    {XML_PATH}")
        print(f"  Modified XML:   {modified_xml}")
    print("=" * 60)


def _run_viz_validation(args: argparse.Namespace) -> None:
    """Visually validate a previously exported modified URDF."""
    if args.model:
        modified_path = args.model
    else:
        modified_path = _select_modified_urdf(URDF_STEM)
    print(f"\nValidating modified URDF: {modified_path}")
    print(f"  Against nominal: {args.urdf}")

    comp = URDFComparison(str(args.urdf), modified_path)
    show_validation(comp)


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full calibration → export → verify → visualise pipeline."""
    # 切换到脚本所在目录,保证相对路径能找到
    import os
    os.chdir(Path(__file__).resolve().parent)
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate input files
    urdf_path = Path(args.urdf)
    config_path = Path(args.config)
    if not urdf_path.exists():
        print(f"Error: URDF not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # ── MODE: --update-model ──
        if args.update_model:
            _run_update_model(args)
            return

        # ── MODE: --viz-validation ──
        if args.viz_validation:
            _run_viz_validation(args)
            return

        # Determine which steps to run
        if args.interactive:
            if not sys.stdin.isatty():
                print(
                    "Warning: --interactive with non-TTY stdin. "
                    "Running full pipeline."
                )
                steps = ["calibrate", "export", "verify", "viz"]
            else:
                steps = _select_steps()
        else:
            steps = ["calibrate", "export", "verify", "viz"]
            if args.calibrate_only:
                steps = ["calibrate"]

        # Auto-enable dependencies
        if "viz" in steps and "export" not in steps:
            print("Note: 'viz' requires a modified URDF. Including 'export' + 'verify'.")
            steps.extend(["export", "verify"])

        # Validate config (needed by calibration step)
        if "calibrate" in steps and not config_path.exists():
            print(f"Error: Config not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        # ── Phase 1: Calibration ──
        result_x = None
        param_names = None
        if "calibrate" in steps:
            result_x, param_names, saved_path = _run_calibration(
                str(urdf_path),
                str(config_path),
                plot=not args.no_plot,
                verbose=args.verbose,
                validation_data=args.validation_data,
            )
            if args.calibrate_only:
                print(
                    f"\nTip: Run `python calibration.py --update-model` "
                    f"to export URDF and verify FK."
                )
                return
        else:
            # Load latest saved results if needed for export
            if "export" in steps or "verify" in steps or "viz" in steps:
                npz_path = _select_npz()
                print(f"Loading calibration results from: {npz_path}")
                data = np.load(npz_path)
                result_x = data["result"]
                param_names = list(data["param_names"])

        params = dict(zip(param_names, result_x))
        print(f"\nLoaded {len(params)} calibration parameters.")

        # ── Phase 2: Export + verify ──
        comp = None
        modified_urdf = None
        modified_xml = None
        if "export" in steps and params is not None:
            modified_urdf, modified_xml, comp, errors = export_with_verification(
                params,
                str(urdf_path),
                output_path=args.output,
                nominal_xml=XML_PATH,
                verbose=args.verbose,
            )

        # ── Phase 3: Visual validation ──
        if "viz" in steps and comp is not None:
            show_validation(comp)

        print("\n" + "=" * 60)
        if result_x is not None:
            if modified_urdf:
                print("Calibration + export complete.")
                print(f"  Nominal URDF:  {urdf_path}")
                print(f"  Modified URDF: {modified_urdf}")
                if modified_xml:
                    print(f"  Nominal XML:   {XML_PATH}")
                    print(f"  Modified XML:  {modified_xml}")
            elif args.calibrate_only:
                print("Calibration results saved.")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
