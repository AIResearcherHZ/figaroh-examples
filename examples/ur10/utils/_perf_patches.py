# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0

"""figaroh 最优轨迹热点的运行时性能补丁。
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

import figaroh.utils.cubic_spline as _cs


def _fast_calc_torque(N, robot, q, v, a):
    """逆动力学:每个采样点只调用一次 RNEA。"""
    tau = np.zeros(robot.model.nv * N)
    for i in range(N):
        tau_i = pin.rnea(robot.model, robot.data, q[i, :], v[i, :], a[i, :])
        for j in range(robot.model.nv):
            tau[j * N + i] = tau_i[j]
    return tau


def _fast_check_cfg_constraints(self, q, v=None, tau=None, soft_lim=0):
    """可行性判断:越界返回 True,否则 False;越界仅 DEBUG 打印。"""
    m = self.rmodel

    # 位置限位
    for j in self.act_idxq:
        delta = soft_lim * abs(m.upperPositionLimit[j] - m.lowerPositionLimit[j])
        if np.any(q[:, j] > m.upperPositionLimit[j] - delta) or np.any(
            q[:, j] < m.lowerPositionLimit[j] + delta
        ):
            _cs.logger.debug("Joint position idx_q %d limits violated!", j)
            _cs.logger.debug("FAILED to generate a feasible cubic spline")
            return True

    # 速度限位
    if v is not None:
        for j in self.act_idxv:
            if np.any(np.abs(v[:, j]) > (1 - soft_lim) * abs(m.velocityLimit[j])):
                _cs.logger.debug("Joint vel idx_v %d limits violated!", j)
                _cs.logger.debug("FAILED to generate a feasible cubic spline")
                return True

    # 力矩限位
    if tau is not None:
        for j in self.act_idxv:
            if np.any(np.abs(tau[:, j]) > (1 - soft_lim) * abs(m.effortLimit[j])):
                _cs.logger.debug("Joint effort idx_v %d limits violated!", j)
                _cs.logger.debug("FAILED to generate a feasible cubic spline")
                return True

    _cs.logger.debug(
        "SUCCEEDED to generate waypoints for a feasible initial cubic spline"
    )
    return False


# IPOPT 迭代上限(原库内为 200)。轨迹优化通常几十次后条件数改善很小,
# 降低上限可明显提速且对结果影响很小。仅作用于轨迹问题。
_TRAJ_MAX_ITERS = 50


def _patch_ipopt_iterations() -> None:
    """限制轨迹 IPOPT 的最大迭代数(只对 BaseTrajectoryIPOPTProblem 生效)。"""
    from figaroh.tools.robotipopt import RobotIPOPTSolver
    from figaroh.optimal.base_optimal_trajectory import BaseTrajectoryIPOPTProblem

    if getattr(RobotIPOPTSolver, "_maxiter_patched", False):
        return
    _orig_init = RobotIPOPTSolver.__init__

    def _init(self, problem, config=None):
        _orig_init(self, problem, config)
        # 仅在轨迹优化问题上收紧迭代上限,其他用途(标定等)不动
        if isinstance(problem, BaseTrajectoryIPOPTProblem) and self.config is not None:
            if self.config.max_iterations > _TRAJ_MAX_ITERS:
                self.config.max_iterations = _TRAJ_MAX_ITERS

    RobotIPOPTSolver.__init__ = _init
    RobotIPOPTSolver._maxiter_patched = True


def apply() -> None:
    """应用全部运行时补丁(可重复调用)。"""
    # 补丁类方法(各处通过 self.CB.check_cfg_constraints 调用)。
    _cs.CubicSpline.check_cfg_constraints = _fast_check_cfg_constraints

    # 定义处和各模块已按名导入的引用都要替换。
    _cs.calc_torque = _fast_calc_torque
    try:
        import figaroh.optimal.base_optimal_trajectory as _bot

        _bot.calc_torque = _fast_calc_torque
    except Exception:  # 防御性处理
        pass
    try:
        import figaroh.optimal.contraints as _con

        _con.calc_torque = _fast_calc_torque
    except Exception:  # 防御性处理
        pass

    # 收紧轨迹 IPOPT 的迭代上限
    try:
        _patch_ipopt_iterations()
    except Exception:  # 防御性处理
        pass


apply()
