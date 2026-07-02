import dataclasses
import time

import numpy as np
import picos as pc
import pinocchio as pin

import figaroh.identification.base_identification as _bi
import figaroh.identification.identification_tools as _it
import figaroh.identification.parameter as _param
import figaroh.optimal.base_optimal_trajectory as _bot
import figaroh.optimal.base_parameter as _bp
import figaroh.optimal.contraints as _con
import figaroh.utils.cubic_spline as _cs
from figaroh.identification import physical_consistency as _phc
from figaroh.optimal.base_optimal_trajectory import BaseTrajectoryIPOPTProblem
from figaroh.tools.robotipopt import RobotIPOPTSolver

_TRAJ_MAX_ITERS = 50


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
            return True

    if v is not None:
        for j in self.act_idxv:
            if np.any(np.abs(v[:, j]) > (1 - soft_lim) * abs(m.velocityLimit[j])):
                return True

    if tau is not None:
        for j in self.act_idxv:
            if np.any(np.abs(tau[:, j]) > (1 - soft_lim) * abs(m.effortLimit[j])):
                return True

    return False


_orig_solver_init = RobotIPOPTSolver.__init__


def _solver_init(self, problem, config=None):
    _orig_solver_init(self, problem, config)
    if isinstance(problem, BaseTrajectoryIPOPTProblem) and self.config is not None:
        if self.config.max_iterations > _TRAJ_MAX_ITERS:
            self.config.max_iterations = _TRAJ_MAX_ITERS


def _get_standard_parameters_fixed(model, identif_config=None):
    inertial_params = [
        "m", "mx", "my", "mz",
        "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz",
    ]

    params: list = []
    phi: list = []
    for jid in range(1, model.njoints):
        jname = model.names[jid]
        pinocchio_params = model.inertias[jid].toDynamicParameters()
        for param_name in inertial_params:
            params.append(f"{param_name}_{jname}")
        phi.extend(pinocchio_params)

    return dict(zip(params, phi))


def _prepare_undecimated_data_fixed(self, regressor_reduced):
    tau = np.asarray(self.processed_data["torques"]).flatten(order="F")
    return tau, regressor_reduced


def _project_p10_lmi_fixed(
    p10_hat,
    *,
    mass_min=1e-6,
    psd_eig_tol=-1e-10,
    weights=None,
    solver="cvxopt",
    verbose=False,
    max_seconds=None,
    mass_bounds=None,
    com_bounds=None,
):
    p10_hat = np.asarray(p10_hat, dtype=float).reshape(10)
    if weights is None:
        w = _phc._auto_weights(p10_hat)
    else:
        w = np.asarray(weights, dtype=float).reshape(10)

    problem = pc.Problem()
    m = pc.RealVariable("m", 1)
    h = pc.RealVariable("h", 3)
    sigma = pc.SymmetricVariable("sigma", 3)
    P = pc.block([[sigma, h], [h.T, m]])
    problem.add_constraint(m >= mass_min)
    problem.add_constraint(P >> 0)

    if mass_bounds is not None:
        problem.add_constraint(m >= mass_bounds[0])
        problem.add_constraint(m <= mass_bounds[1])
    if com_bounds:
        axis_idx = {"x": 0, "y": 1, "z": 2}
        for axis, (h_lo, h_hi) in com_bounds.items():
            k = axis_idx[axis]
            problem.add_constraint(h[k] >= h_lo)
            problem.add_constraint(h[k] <= h_hi)

    tr_sigma = pc.trace(sigma)

    def _I_expr(r, c):
        e = -sigma[r, c]
        if r == c:
            e = e + tr_sigma
        return e

    obj = pc.SquaredNorm(float(w[0]) * (m - float(p10_hat[0])))
    for k in range(3):
        obj = obj + (float(w[1 + k]) * (h[k] - float(p10_hat[1 + k]))) ** 2
    for r, c, idx in [
        (0, 0, 4),
        (0, 1, 5),
        (1, 1, 6),
        (0, 2, 7),
        (1, 2, 8),
        (2, 2, 9),
    ]:
        obj = obj + (float(w[idx]) * (_I_expr(r, c) - float(p10_hat[idx]))) ** 2
    problem.minimize = obj

    solve_kwargs = {"solver": solver, "verbosity": int(verbose)}
    if max_seconds is not None:
        solve_kwargs["max_seconds"] = float(max_seconds)

    t_start = time.perf_counter()
    problem.solve(**solve_kwargs)
    t_elapsed = time.perf_counter() - t_start

    m_val = float(m.value)
    h_val = np.asarray(h.value).reshape(3)
    S = np.asarray(sigma.value).reshape(3, 3)
    S = 0.5 * (S + S.T)
    I = np.trace(S) * np.eye(3) - S

    p10_proj = np.array([
        m_val, h_val[0], h_val[1], h_val[2],
        I[0, 0], I[0, 1], I[1, 1], I[0, 2], I[1, 2], I[2, 2],
    ])

    feas = _phc.check_p10_feasibility(
        p10_proj, mass_min=mass_min, psd_eig_tol=min(psd_eig_tol, -1e-8)
    )
    report = dataclasses.replace(
        feas,
        status="projected" if feas.status == "feasible" else "infeasible",
        solver=solver,
        objective=float(problem.value) if problem.value is not None else None,
        runtime=t_elapsed,
    )
    return p10_proj, report


def apply() -> None:
    _param.get_standard_parameters = _get_standard_parameters_fixed
    _it.get_standard_parameters = _get_standard_parameters_fixed
    _bi.get_standard_parameters = _get_standard_parameters_fixed
    _bp.get_standard_parameters = _get_standard_parameters_fixed

    _bi.BaseIdentification._prepare_undecimated_data = _prepare_undecimated_data_fixed
    _phc.project_p10_lmi = _project_p10_lmi_fixed

    _cs.CubicSpline.check_cfg_constraints = _fast_check_cfg_constraints
    _cs.calc_torque = _fast_calc_torque
    _bot.calc_torque = _fast_calc_torque
    _con.calc_torque = _fast_calc_torque

    RobotIPOPTSolver.__init__ = _solver_init


apply()
