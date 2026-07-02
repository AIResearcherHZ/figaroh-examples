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
import datetime
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.ur10.utils.ur10_tools import UR10Calibration
from figaroh.tools.robot import load_robot
from figaroh.tools.urdf_exporter import (
    export_urdf,
    frame_settings_doc,
)
from figaroh.tools.export_validation import URDFComparison

logger = logging.getLogger(__name__)

DATA_DIR = "data/calibration"
URDF_STEM = "ur10_robot"
XML_PATH = "../../models/ur_description/ur10.xml"


def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
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
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _parse_float(val) -> float:
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
    import xml.etree.ElementTree as ET

    nominal = Path(nominal_xml_path)
    if not nominal.exists():
        raise FileNotFoundError(f"MJCF not found: {nominal}")

    if output_path is None:
        output_path = str(nominal)
    out = Path(output_path)

    tree = ET.parse(str(nominal))
    root = tree.getroot()

    body_map: dict[str, ET.Element] = {}
    for body in root.iter("body"):
        name = body.get("name")
        if name:
            body_map[name] = body

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

    placement: dict[str, list[float]] = {}
    for name, val in params.items():
        for axis, idx in [("d_px", 0), ("d_py", 1), ("d_pz", 2),
                          ("d_phix", 3), ("d_phiy", 4), ("d_phiz", 5)]:
            if name.startswith(f"{axis}_"):
                target = name[len(axis) + 1:]
                if target:
                    placement.setdefault(target, [0.0] * 6)[idx] = _parse_float(val)
                break

    mass: dict[str, float] = {}
    moments: dict[str, list[float]] = {}
    inertia: dict[str, np.ndarray] = {}
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

    friction: dict[str, dict] = {}
    for name, val in params.items():
        v = _parse_float(val)
        if name.startswith("fv_"):
            friction.setdefault(name[3:], {})["damping"] = v
        elif name.startswith("fs_"):
            friction.setdefault(name[3:], {})["frictionloss"] = v
        elif name.startswith("Ia_"):
            friction.setdefault(name[3:], {})["armature"] = v

    applied = 0

    for target, deltas in placement.items():
        body = _get_body(target)
        if body is None:
            if verbose:
                logger.warning("XML body '%s' not found, skipping", target)
            continue
        cur_pos = [float(x) for x in body.get("pos", "0 0 0").split()]
        while len(cur_pos) < 3:
            cur_pos.append(0.0)
        new_pos = [cur_pos[i] + deltas[i] for i in range(3)]
        body.set("pos", " ".join(_fmt_xml(x) for x in new_pos))
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

    all_bodies = set(mass) | set(moments) | set(inertia)
    for bname in all_bodies:
        body = _get_body(bname)
        if body is None:
            if verbose:
                logger.warning("XML body '%s' not found, skipping", bname)
            continue
        ine = _get_or_create_inertial(body)

        m = mass.get(bname)
        if m is None:
            m = _parse_float(ine.get("mass", 0.0))
        else:
            ine.set("mass", _fmt_xml(m))

        if bname in moments and m and m != 0:
            mx, my, mz = moments[bname]
            com = np.array([mx / m, my / m, mz / m])
            ine.set("pos", " ".join(_fmt_xml(x) for x in com))
        else:
            cur = [float(x) for x in ine.get("pos", "0 0 0").split()]
            while len(cur) < 3:
                cur.append(0.0)
            com = np.array(cur[:3])

        if bname in inertia:
            I_O = inertia[bname]
            c = com
            I_C = I_O - m * (float(c @ c) * np.eye(3) - np.outer(c, c))
            Ixx, Iyy, Izz = I_C[0, 0], I_C[1, 1], I_C[2, 2]
            Ixy, Ixz, Iyz = I_C[0, 1], I_C[0, 2], I_C[1, 2]
            for attr in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
                ine.attrib.pop(attr, None)
            if abs(Ixy) < 1e-12 and abs(Ixz) < 1e-12 and abs(Iyz) < 1e-12:
                ine.set("diaginertia",
                        " ".join(_fmt_xml(x) for x in [Ixx, Iyy, Izz]))
                ine.attrib.pop("fullinertia", None)
            else:
                ine.set("fullinertia",
                        " ".join(_fmt_xml(x)
                                 for x in [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]))
                ine.attrib.pop("diaginertia", None)
        applied += 1
        if verbose:
            logger.info("XML body '%s' inertia updated", bname)

    for jname, attrs in friction.items():
        body = _get_body(jname)
        if body is None:
            if verbose:
                logger.warning("XML body '%s' not found, skipping", jname)
            continue
        jt = _get_joint(body, jname)
        if jt is None:
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


def export_urdf_dynamics(
    nominal_urdf_path: str,
    params: dict,
    *,
    output_path: str | None = None,
    verbose: bool = False,
) -> str:
    import xml.etree.ElementTree as ET

    nominal = Path(nominal_urdf_path)
    if not nominal.exists():
        raise FileNotFoundError(f"URDF not found: {nominal}")
    out = Path(output_path) if output_path else nominal

    mass: dict[str, float] = {}
    moments: dict[str, list[float]] = {}
    inertia: dict[str, np.ndarray] = {}
    friction: dict[str, dict] = {}
    rest: dict = {}
    for name, val in params.items():
        v = _parse_float(val)
        if name.startswith("m_") and not name.startswith(("mx_", "my_", "mz_")):
            mass[name[2:]] = v
        elif name.startswith("mx_"):
            moments.setdefault(name[3:], [0.0, 0.0, 0.0])[0] = v
        elif name.startswith("my_"):
            moments.setdefault(name[3:], [0.0, 0.0, 0.0])[1] = v
        elif name.startswith("mz_"):
            moments.setdefault(name[3:], [0.0, 0.0, 0.0])[2] = v
        elif name[:4] in ("Ixx_", "Iyy_", "Izz_", "Ixy_", "Ixz_", "Iyz_"):
            key, tgt = name[:3], name[4:]
            I = inertia.setdefault(tgt, np.zeros((3, 3)))
            r, c = {"Ixx": (0, 0), "Iyy": (1, 1), "Izz": (2, 2),
                    "Ixy": (0, 1), "Ixz": (0, 2), "Iyz": (1, 2)}[key]
            I[r, c] = v
            I[c, r] = v
        elif name.startswith("fv_"):
            friction.setdefault(name[3:], {})["damping"] = v
        elif name.startswith("fs_"):
            friction.setdefault(name[3:], {})["friction"] = v
        elif name.startswith("Ia_"):
            friction.setdefault(name[3:], {})["armature"] = v
        else:
            rest[name] = val

    tree = ET.parse(str(nominal))
    root = tree.getroot()

    links = {lk.get("name"): lk for lk in root.findall(".//link")}
    joint_child: dict[str, ET.Element] = {}
    joint_elem: dict[str, ET.Element] = {}
    for jt in root.findall(".//joint"):
        jname = jt.get("name")
        child = jt.find("child")
        if jname and child is not None:
            joint_elem[jname] = jt
            lk = links.get(child.get("link", ""))
            if lk is not None:
                joint_child[jname] = lk

    def _get_or_create(parent: ET.Element, tag: str) -> ET.Element:
        e = parent.find(tag)
        if e is None:
            e = ET.SubElement(parent, tag)
        return e

    applied = 0

    for jname in set(mass) | set(moments) | set(inertia):
        lk = joint_child.get(jname)
        if lk is None:
            logger.warning("URDF joint '%s' (child link) not found, skipping",
                           jname)
            continue
        ine = _get_or_create(lk, "inertial")
        m_el = _get_or_create(ine, "mass")
        m = mass.get(jname)
        if m is None:
            m = _parse_float(m_el.get("value", 0.0))
        else:
            m_el.set("value", _fmt_xml(m))

        origin = _get_or_create(ine, "origin")
        if jname in moments and m and m != 0:
            mx, my, mz = moments[jname]
            com = np.array([mx / m, my / m, mz / m])
            origin.set("xyz", " ".join(_fmt_xml(x) for x in com))
        else:
            cur = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            while len(cur) < 3:
                cur.append(0.0)
            com = np.array(cur[:3])

        if jname in inertia:
            I_O = inertia[jname]
            c = com
            I_C = I_O - m * (float(c @ c) * np.eye(3) - np.outer(c, c))
            origin.set("rpy", "0 0 0")
            i_el = _get_or_create(ine, "inertia")
            i_el.set("ixx", _fmt_xml(I_C[0, 0]))
            i_el.set("ixy", _fmt_xml(I_C[0, 1]))
            i_el.set("ixz", _fmt_xml(I_C[0, 2]))
            i_el.set("iyy", _fmt_xml(I_C[1, 1]))
            i_el.set("iyz", _fmt_xml(I_C[1, 2]))
            i_el.set("izz", _fmt_xml(I_C[2, 2]))
        applied += 1
        if verbose:
            logger.info("URDF link '%s' (joint '%s') inertial updated",
                        lk.get("name"), jname)

    for jname, attrs in friction.items():
        jt = joint_elem.get(jname)
        if jt is None:
            logger.warning("URDF joint '%s' not found, skipping", jname)
            continue
        dyn = _get_or_create(jt, "dynamics")
        for attr, v in attrs.items():
            dyn.set(attr, _fmt_xml(v))
        applied += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), xml_declaration=True, encoding="utf-8")
    print(f"URDF exported: {out}  ({applied} bodies/joints updated)")

    if rest:
        export_urdf(str(out), rest, output_path=str(out), verbose=verbose)

    return str(out.resolve())


def _fmt_xml(v: float) -> str:
    if v == 0.0:
        return "0"
    s = f"{v:.10g}"
    if "e" in s or "E" in s:
        s = f"{v:.10f}".rstrip("0").rstrip(".")
    return s


def _timestamp_str() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _discover_npz_files(data_dir: str = DATA_DIR) -> list[Path]:
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
    files = sorted(Path("urdf").glob(f"{stem}_modified_*.urdf"))
    if not files:
        legacy = Path(f"urdf/{stem}_modified.urdf")
        if legacy.exists():
            files = [legacy]
    return files


def _select_npz(data_dir: str = DATA_DIR) -> str:
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


def _select_steps() -> list[str]:
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


def parse_args() -> argparse.Namespace:
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
        default="../../models/ur_description/urdf/ur10_robot.urdf",
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


def _run_calibration(
    urdf_path: str,
    config_path: str,
    *,
    plot: bool = True,
    verbose: bool = False,
    validation_data: str | None = None,
) -> tuple[np.ndarray, list[str], str]:
    ur10 = load_robot(urdf_path, package_dirs="../../models", load_by_urdf=True)
    ur10_calib = UR10Calibration(ur10, config_path)
    ur10_calib.calib_config["known_baseframe"] = False
    ur10_calib.calib_config["known_tipframe"] = False
    if validation_data:
        ur10_calib.calib_config["validation_data_file"] = validation_data
    ur10_calib.initialize()
    result = ur10_calib.solve(plotting=plot, enable_logging=verbose)
    param_names = ur10_calib.calib_config["param_name"]

    os.makedirs(DATA_DIR, exist_ok=True)
    ts = _timestamp_str()
    saved_path = os.path.join(DATA_DIR, f"calibration_results_{ts}.npz")
    np.savez(saved_path, result=result.x, param_names=param_names)
    print(f"Calibration results saved to {saved_path}")

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
    nominal_path = Path(nominal_urdf)

    if output_path is None:
        output_path = str(nominal_path)

    modified_path = export_urdf(
        str(nominal_path),
        params,
        output_path=str(output_path),
        verbose=verbose,
    )

    modified_xml = None
    if nominal_xml and Path(nominal_xml).exists():
        modified_xml = export_xml(
            str(nominal_xml),
            params,
            output_path=str(xml_output_path) if xml_output_path else None,
            verbose=verbose,
        )

    frame_settings_doc(calibration_type=calibration_type, verbose=verbose)

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


def show_validation(comp: URDFComparison):
    try:
        import viser
    except ImportError:
        print("  viser not installed; skipping visual validation.")
        return

    comp.show_interactive_validation(n_trajectory=50, port=8080)


def _run_update_model(args: argparse.Namespace) -> None:
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
    if args.model:
        modified_path = args.model
    else:
        modified_path = _select_modified_urdf(URDF_STEM)
    print(f"\nValidating modified URDF: {modified_path}")
    print(f"  Against nominal: {args.urdf}")

    comp = URDFComparison(str(args.urdf), modified_path)
    show_validation(comp)


def main() -> None:
    import os
    os.chdir(Path(__file__).resolve().parent)
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    urdf_path = Path(args.urdf)
    config_path = Path(args.config)
    if not urdf_path.exists():
        print(f"Error: URDF not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.update_model:
            _run_update_model(args)
            return

        if args.viz_validation:
            _run_viz_validation(args)
            return

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

        if "viz" in steps and "export" not in steps:
            print("Note: 'viz' requires a modified URDF. Including 'export' + 'verify'.")
            steps.extend(["export", "verify"])

        if "calibrate" in steps and not config_path.exists():
            print(f"Error: Config not found: {config_path}", file=sys.stderr)
            sys.exit(1)

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
            if "export" in steps or "verify" in steps or "viz" in steps:
                npz_path = _select_npz()
                print(f"Loading calibration results from: {npz_path}")
                data = np.load(npz_path)
                result_x = data["result"]
                param_names = list(data["param_names"])

        params = dict(zip(param_names, result_x))
        print(f"\nLoaded {len(params)} calibration parameters.")

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
