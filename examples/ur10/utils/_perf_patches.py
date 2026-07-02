# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0


from __future__ import annotations

import numpy as np
import pinocchio as pin

import figaroh.utils.cubic_spline as _cs


def _fast_calc_torque(N, robot, q, v, a):
    tau = np.zeros(robot.model.nv * N)
    for i in range(N):
        tau_i = pin.rnea(robot.model, robot.data, q[i, :], v[i, :], a[i, :])
        for j in range(robot.model.nv):
            tau[j * N + i] = tau_i[j]
    return tau


def _fast_check_cfg_constraints(self, q, v=None, tau=None, soft_lim=0):
    m = self.rmodel

    for j in self.act_idxq:
        delta = soft_lim * abs(m.upperPositionLimit[j] - m.lowerPositionLimit[j])
        if np.any(q[:, j] > m.upperPositionLimit[j] - delta) or np.any(
            q[:, j] < m.lowerPositionLimit[j] + delta
        ):
            _cs.logger.debug("Joint position idx_q %d limits violated!", j)
            _cs.logger.debug("FAILED to generate a feasible cubic spline")
            return True

    if v is not None:
        for j in self.act_idxv:
            if np.any(np.abs(v[:, j]) > (1 - soft_lim) * abs(m.velocityLimit[j])):
                _cs.logger.debug("Joint vel idx_v %d limits violated!", j)
                _cs.logger.debug("FAILED to generate a feasible cubic spline")
                return True

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


_TRAJ_MAX_ITERS = 50


def _patch_ipopt_iterations() -> None:
    from figaroh.tools.robotipopt import RobotIPOPTSolver
    from figaroh.optimal.base_optimal_trajectory import BaseTrajectoryIPOPTProblem

    if getattr(RobotIPOPTSolver, "_maxiter_patched", False):
        return
    _orig_init = RobotIPOPTSolver.__init__

    def _init(self, problem, config=None):
        _orig_init(self, problem, config)
        if isinstance(problem, BaseTrajectoryIPOPTProblem) and self.config is not None:
            if self.config.max_iterations > _TRAJ_MAX_ITERS:
                self.config.max_iterations = _TRAJ_MAX_ITERS

    RobotIPOPTSolver.__init__ = _init
    RobotIPOPTSolver._maxiter_patched = True


def _get_standard_parameters_fixed(model, identif_config=None):
    inertial_params = [
        "m", "mx", "my", "mz",
        "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz",
    ]

    params: list = []
    phi: list = []
    assert len(model.inertias) == model.njoints, \
        "Inertia count mismatch with joints"
    for jid in range(1, model.njoints):
        jname = model.names[jid]
        pinocchio_params = model.inertias[jid].toDynamicParameters()
        for param_name in inertial_params:
            params.append(f"{param_name}_{jname}")
        phi.extend(pinocchio_params)

    return dict(zip(params, phi))


def _patch_standard_parameters() -> None:
    import importlib

    import figaroh.identification.parameter as _param

    if getattr(_param.get_standard_parameters, "_offbyone_fixed", False):
        return
    _get_standard_parameters_fixed._offbyone_fixed = True

    _param.get_standard_parameters = _get_standard_parameters_fixed
    for mod_name in (
        "figaroh.identification.identification_tools",
        "figaroh.identification.base_identification",
        "figaroh.optimal.base_parameter",
    ):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "get_standard_parameters"):
                mod.get_standard_parameters = _get_standard_parameters_fixed
        except Exception:
            pass


def apply() -> None:
    try:
        _patch_standard_parameters()
    except Exception:
        pass

    _cs.CubicSpline.check_cfg_constraints = _fast_check_cfg_constraints

    _cs.calc_torque = _fast_calc_torque
    try:
        import figaroh.optimal.base_optimal_trajectory as _bot

        _bot.calc_torque = _fast_calc_torque
    except Exception:
        pass
    try:
        import figaroh.optimal.contraints as _con

        _con.calc_torque = _fast_calc_torque
    except Exception:
        pass

    try:
        _patch_ipopt_iterations()
    except Exception:
        pass


apply()
