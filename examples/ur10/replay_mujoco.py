import argparse
import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

Q_CSV = "data/identification_q_simulation.csv"
TAU_CSV = "data/identification_tau_simulation.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="用 MuJoCo 生成 UR10 辨识数据(q + tau)")
    ap.add_argument(
        "--model", default="../../models/ur_description/ur10.xml",
        help="MJCF 模型路径",
    )
    ap.add_argument(
        "--csv", default="results/ur10_optimal_trajectory.csv",
        help="输入轨迹 CSV(含 time, q1..q6, dq1..dq6, ddq1..ddq6)",
    )
    ap.add_argument(
        "--q-out", default=Q_CSV,
        help="关节位置输出 CSV 路径",
    )
    ap.add_argument(
        "--tau-out", default=TAU_CSV,
        help="关节力矩输出 CSV 路径",
    )
    ap.add_argument(
        "--viewer", action="store_true",
        help="生成数据后打开 MuJoCo 可视化回放",
    )
    return ap.parse_args()


def generate_data(
    model_path: str, csv_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    df = pd.read_csv(csv_path)
    q = df[[f"q{i}" for i in range(1, 7)]].to_numpy()
    dq = df[[f"dq{i}" for i in range(1, 7)]].to_numpy()
    ddq = df[[f"ddq{i}" for i in range(1, 7)]].to_numpy()

    qadr = [model.joint(n).qposadr[0] for n in JOINT_NAMES]
    vadr = [model.joint(n).dofadr[0] for n in JOINT_NAMES]

    N = len(q)
    tau = np.zeros((N, 6))

    for i in range(N):
        data.qpos[qadr] = q[i]
        data.qvel[vadr] = dq[i]
        data.qacc[vadr] = ddq[i]
        mujoco.mj_inverse(model, data)
        tau[i] = data.qfrc_inverse[vadr]

    return q, dq, ddq, tau


def save_csv(
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    tau: np.ndarray,
    q_path: str,
    tau_path: str,
) -> None:
    Path(q_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tau_path).parent.mkdir(parents=True, exist_ok=True)

    cols = {}
    for j in range(6):
        cols[f"q{j + 1}"] = q[:, j]
    for j in range(6):
        cols[f"dq{j + 1}"] = dq[:, j]
        cols[f"ddq{j + 1}"] = ddq[:, j]
    q_df = pd.DataFrame(cols)
    q_df.to_csv(q_path, index=False)

    tau_df = pd.DataFrame(tau, columns=[f"tau{i+1}" for i in range(6)])
    tau_df.to_csv(tau_path, index=False)

    print(f"q   -> {q_path}  ({len(q)} 行)")
    print(f"tau -> {tau_path}  ({len(tau)} 行)")


def replay_viewer(model_path: str, q: np.ndarray) -> None:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    qadr = [model.joint(n).qposadr[0] for n in JOINT_NAMES]

    print("打开 MuJoCo 可视化(关闭窗口退出)...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(len(q)):
            if not viewer.is_running():
                break
            data.qpos[qadr] = q[i]
            mujoco.mj_forward(model, data)
            viewer.sync()
            import time
            time.sleep(model.opt.timestep)


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    args = parse_args()

    print(f"模型: {args.model}")
    print(f"轨迹: {args.csv}")

    q, dq, ddq, tau = generate_data(args.model, args.csv)
    save_csv(q, dq, ddq, tau, args.q_out, args.tau_out)

    print(f"\n力矩统计(min / max / mean):")
    for j in range(6):
        print(f"  tau{j+1}: {tau[:, j].min():12.4f} / {tau[:, j].max():12.4f} / {tau[:, j].mean():12.4f}")

    if args.viewer:
        replay_viewer(args.model, q)


if __name__ == "__main__":
    main()