# Copyright [2021-2025] Thanh Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# See http://www.apache.org/licenses/LICENSE-2.0

"""在 MuJoCo 中回放 UR10 最优轨迹 CSV。

默认循环播放,并用 rich 打印回放进度条。

用法::

    python replay_mujoco.py                 # 循环回放默认 CSV
    python replay_mujoco.py --no-loop       # 只播放一次
    python replay_mujoco.py --speed 2       # 2 倍速
    python replay_mujoco.py --csv 其它.csv

CSV 需含列 time, q1..q6(由 optimal_trajectory.py 生成)。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
)

# CSV 里 q1..q6 对应的关节名(顺序即 MuJoCo qpos 映射依据)
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MuJoCo 回放 UR10 轨迹")
    ap.add_argument(
        "--model",
        default="../../models/ur_description/ur10.xml",
        help="MJCF 模型路径",
    )
    ap.add_argument(
        "--csv",
        default="results/ur10_optimal_trajectory.csv",
        help="轨迹 CSV 路径",
    )
    ap.add_argument("--speed", type=float, default=1.0, help="回放倍速")
    ap.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="只播放一次(默认循环播放)",
    )
    ap.set_defaults(loop=True)
    return ap.parse_args()


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    args = parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    df = pd.read_csv(args.csv)
    Q = df[[f"q{i}" for i in range(1, 7)]].to_numpy()
    t = df["time"].to_numpy()
    qadr = [model.joint(n).qposadr[0] for n in JOINT_NAMES]

    # CSV 采样率通常很低(如 10Hz),逐帧播放只有 10fps 明显卡顿。
    # 这里改为时间驱动 + 关节空间线性插值,以固定 60fps 平滑渲染。
    n = len(Q)
    diffs = np.diff(t)
    dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.1
    total = max((n - 1) * dt, 1e-6)
    frame_dt = 1.0 / 60.0

    print(f"模型: {args.model}")
    print(f"轨迹: {args.csv}  ({n} 帧 @ {1 / dt:.0f}Hz, 单次时长 {total:.1f}s)")
    print(f"回放: {'循环' if args.loop else '单次'}, {args.speed}x —— 关闭窗口退出。")

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("{task.fields[pos]:.1f}/{task.fields[tot]:.1f}s"),
        transient=False,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer, progress:
        cycle = 1
        task = progress.add_task(
            f"回放 #{cycle}", total=total, pos=0.0, tot=total
        )
        start = time.time()
        while viewer.is_running():
            elapsed = (time.time() - start) * args.speed
            if elapsed >= total:
                if args.loop:
                    cycle += 1
                    start = time.time()
                    elapsed = 0.0
                    progress.reset(
                        task, description=f"回放 #{cycle}", pos=0.0, tot=total
                    )
                else:
                    elapsed = total

            # 浮点帧位置 -> 相邻两帧线性插值
            if n > 1:
                fpos = elapsed / dt
                i = min(int(fpos), n - 2)
                frac = fpos - i
                q = Q[i] * (1.0 - frac) + Q[i + 1] * frac
            else:
                q = Q[0]

            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            viewer.sync()
            progress.update(task, completed=min(elapsed, total), pos=min(elapsed, total))

            if not args.loop and elapsed >= total:
                break
            time.sleep(frame_dt)

        # 回放结束后保持窗口,直到用户关闭
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
