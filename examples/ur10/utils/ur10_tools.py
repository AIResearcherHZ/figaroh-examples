from __future__ import annotations

import os
import yaml
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List

from figaroh.calibration.calibration_tools import (
    load_data,
    calc_updated_fkm,
)

from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.identification.base_identification import BaseIdentification
from figaroh.optimal.base_optimal_calibration import BaseOptimalCalibration
from figaroh.utils.error_handling import handle_calibration_errors
from figaroh.identification.identification_tools import (
    get_param_from_yaml as get_identification_param_from_yaml,
)
from figaroh.tools.regressor import (
    build_regressor_basic,
    get_index_eliminate,
    build_regressor_reduced,
)
from figaroh.tools.qrdecomposition import get_baseParams
from figaroh.identification.parameter import get_standard_parameters
from figaroh.optimal.base_optimal_trajectory import (
    BaseOptimalTrajectory,
    BaseTrajectoryIPOPTProblem,
)

from . import _perf_patches

try:
    from .data_processing import DataProcessor
except ImportError:
    print("DataProcessor module not found, using basic data loading methods.")


class UR10Calibration(BaseCalibration):

    @handle_calibration_errors
    def __init__(
        self,
        robot: Any,
        config_file: str = "config/ur10_config.yaml",
        del_list: List[int] | None = None,
    ) -> None:
        super().__init__(robot, config_file, del_list or [])

    def cost_function(self, var: np.ndarray) -> np.ndarray:
        coeff_ = self.calib_config["coeff_regularize"]
        PEEe = calc_updated_fkm(
            self.model, self.data, var, self.q_measured, self.calib_config
        )

        raw_residuals = self._compute_logmap_residuals(
            self.PEE_measured, PEEe
        )

        weighted_residuals = self.apply_measurement_weighting(
            raw_residuals, pos_weight=1.0, orient_weight=0.5
        )

        n_base_params = 6
        n_markers = self.calib_config["NbMarkers"]
        n_tip_params = n_markers * self.calib_config["calibration_index"]
        regularization_params = var[n_base_params:-n_tip_params]
        regularization_residuals = np.sqrt(coeff_) * regularization_params

        res_vect = np.append(weighted_residuals, regularization_residuals)
        return res_vect


class UR10Identification(BaseIdentification):

    def __init__(
        self, robot: Any, config_file: str = "config/ur10_config.yaml"
    ) -> None:
        super().__init__(robot, config_file)
        print("UR10 Dynamic Identification initialized")

    def load_trajectory_data(self) -> Dict[str, np.ndarray]:
        q_df = pd.read_csv("data/identification_q_simulation.csv")
        tau_df = pd.read_csv("data/identification_tau_simulation.csv")

        q = q_df[[f"q{j}" for j in range(1, 7)]].to_numpy()
        dq = q_df[[f"dq{j}" for j in range(1, 7)]].to_numpy()
        ddq = q_df[[f"ddq{j}" for j in range(1, 7)]].to_numpy()
        tau = tau_df.to_numpy()[: len(q)]
        print(f"Loaded {len(q)} samples with exact dq/ddq")

        dt = 0.02
        time_vector = np.arange(len(q)) * dt

        return {
            "timestamps": time_vector.reshape(-1, 1),
            "positions": q,
            "velocities": dq,
            "accelerations": ddq,
            "torques": tau,
        }

    def process_kinematics_data(self, filter_config=None) -> None:
        self.processed_data = {
            "timestamps": self.raw_data["timestamps"],
            "positions": self.raw_data["positions"],
            "velocities": self.raw_data["velocities"],
            "accelerations": self.raw_data["accelerations"],
        }


class UR10OptimalCalibration(BaseOptimalCalibration):

    def __init__(
        self, robot: Any, config_file: str = "config/ur10_config.yaml"
    ) -> None:
        super().__init__(robot, config_file)
        print("UR10 Optimal Calibration initialized")

    def load_candidate_configurations(self) -> None:
        from figaroh.calibration.calibration_tools import get_idxq_from_jname

        if self._sampleConfigs_file is None:
            print(
                "No sample configurations file specified, generating random configurations"
            )
            self._generate_random_configurations()
            return

        try:
            if "csv" in self._sampleConfigs_file:
                self.q_measured, _ = load_data(
                    self._sampleConfigs_file, self.model, self.calib_config, []
                )
                self.calib_config["NbSample"] = len(self.q_measured)
                print(f"Loaded {len(self.q_measured)} configurations from CSV file")

            elif "yaml" in self._sampleConfigs_file:
                with open(self._sampleConfigs_file, "r") as f:
                    configs_data = yaml.load(f, Loader=SafeLoader)

                joint_names = configs_data["calibration_joint_names"]
                joint_configs = configs_data["calibration_joint_configurations"]

                idxq = get_idxq_from_jname(self.model, joint_names)
                q_configs = np.array(joint_configs)

                self.q_measured = np.zeros((len(joint_configs), self.model.nq))
                self.q_measured[:, idxq] = q_configs

                self._configs = configs_data

                self.calib_config["NbSample"] = len(self.q_measured)
                print(f"Loaded {len(self.q_measured)} configurations from YAML file")

            else:
                raise ValueError(f"Unsupported file format: {self._sampleConfigs_file}")

        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Failed to load candidate configurations: {e}")
            print("Generating random configurations instead")
            self._generate_random_configurations()

    def _generate_random_configurations(self) -> None:
        print("Generating random configurations for UR10...")

        q_min = np.array(
            [-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi]
        )
        q_max = np.array([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])

        n_samples = 500
        self.q_measured = np.random.uniform(
            low=q_min, high=q_max, size=(n_samples, len(q_min))
        )

        self.calib_config["NbSample"] = len(self.q_measured)

        joint_names = [f"joint_{i+1}" for i in range(len(q_min))]
        self._configs = {
            "calibration_joint_names": joint_names,
            "calibration_joint_configurations": self.q_measured.tolist(),
        }

        print(f"Generated {len(self.q_measured)} random configurations")


class UR10OptimalTrajectory:

    def __init__(
        self, robot: Any, config_file: str = "config/ur10_config.yaml"
    ) -> None:
        self.robot = robot
        self.model = robot.model
        self.data = robot.data

        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=SafeLoader)

        self.config = config
        self.identif_data = self.config["identification"]
        self.identif_config = get_identification_param_from_yaml(
            robot, self.identif_data
        )

        self.n_waypoints = 10
        self.trajectory_duration = 10.0
        self.dt = 0.01

        print("UR10 Optimal Trajectory initialized")

    def generate_base_parameters(self) -> None:
        q_rand = np.random.uniform(low=-np.pi, high=np.pi, size=(1000, self.model.nq))
        dq_rand = np.random.uniform(low=-10, high=10, size=(1000, self.model.nv))
        ddq_rand = np.random.uniform(low=-10, high=10, size=(1000, self.model.nv))

        W = build_regressor_basic(
            self.robot, q_rand, dq_rand, ddq_rand, self.identif_config
        )

        self.standard_parameter = get_standard_parameters(
            self.model, self.identif_config
        )

        idx_e, params_r = get_index_eliminate(W, self.standard_parameter, 1e-6)
        W_e = build_regressor_reduced(W, idx_e)

        _, params_base, idx_base = get_baseParams(
            W_e, params_r, self.standard_parameter
        )
        self.params_base = params_base
        self.idx_base = idx_base

        print(f"Generated {len(params_base)} base parameters")

    def cubic_spline_trajectory(
        self, waypoints: np.ndarray, total_time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.interpolate import CubicSpline

        n_points = int(total_time / self.dt)
        t = np.linspace(0, total_time, len(waypoints))
        t_eval = np.linspace(0, total_time, n_points)

        trajectories = []
        velocities = []
        accelerations = []

        for joint in range(self.model.nq):
            cs = CubicSpline(t, waypoints[:, joint])
            trajectories.append(cs(t_eval))
            velocities.append(cs(t_eval, nu=1))
            accelerations.append(cs(t_eval, nu=2))

        q = np.array(trajectories).T
        dq = np.array(velocities).T
        ddq = np.array(accelerations).T

        return q, dq, ddq

    def check_constraints(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> bool:
        q_min = -np.pi * np.ones(self.model.nq)
        q_max = np.pi * np.ones(self.model.nq)

        if np.any(q < q_min) or np.any(q > q_max):
            return False

        dq_max = 2.0 * np.ones(self.model.nv)
        if np.any(np.abs(dq) > dq_max):
            return False

        ddq_max = 5.0 * np.ones(self.model.nv)
        if np.any(np.abs(ddq) > ddq_max):
            return False

        return True

    def evaluate_trajectory_quality(
        self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray
    ) -> float:
        W = build_regressor_basic(self.robot, q, dq, ddq, self.identif_config)

        idx_e, params_r = get_index_eliminate(W, self.standard_parameter, 1e-6)
        W_e = build_regressor_reduced(W, idx_e)

        W_base = W_e[:, self.idx_base]

        cond_num = np.linalg.cond(W_base)

        return cond_num

    def solve(self, n_iterations: int = 50) -> dict | None:
        print("Generating optimal trajectory for UR10 identification...")

        self.generate_base_parameters()

        best_trajectory = None
        best_condition_number = float("inf")

        for i in range(n_iterations):
            waypoints = np.random.uniform(
                low=-np.pi, high=np.pi, size=(self.n_waypoints, self.model.nq)
            )

            q, dq, ddq = self.cubic_spline_trajectory(
                waypoints, self.trajectory_duration
            )

            if not self.check_constraints(q, dq, ddq):
                continue

            cond_num = self.evaluate_trajectory_quality(q, dq, ddq)

            if cond_num < best_condition_number:
                best_condition_number = cond_num
                best_trajectory = {
                    "q": q,
                    "dq": dq,
                    "ddq": ddq,
                    "waypoints": waypoints,
                    "condition_number": cond_num,
                }

            if i % 10 == 0:
                print(
                    f"Iteration {i}: best condition number = {best_condition_number:.2e}"
                )

        self.optimal_trajectory = best_trajectory
        print(
            f"Optimal trajectory found with condition number: {best_condition_number:.2e}"
        )

        return best_trajectory

    def plot_results(self) -> None:
        if not hasattr(self, "optimal_trajectory"):
            print("No optimal trajectory results to plot. Run solve() first.")
            return

        traj = self.optimal_trajectory
        time_vector = np.arange(len(traj["q"])) * self.dt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(time_vector, traj["q"])
        axes[0].set_ylabel("Joint Position (rad)")
        axes[0].set_title("UR10 Optimal Trajectory - Joint Positions")
        axes[0].grid(True)
        axes[0].legend([f"Joint {i+1}" for i in range(self.model.nq)])

        axes[1].plot(time_vector, traj["dq"])
        axes[1].set_ylabel("Joint Velocity (rad/s)")
        axes[1].set_title("Joint Velocities")
        axes[1].grid(True)

        axes[2].plot(time_vector, traj["ddq"])
        axes[2].set_ylabel("Joint Acceleration (rad/s²)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Joint Accelerations")
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

        print(f"Trajectory condition number: {traj['condition_number']:.2e}")

    def save_results(self, output_dir: str = "results") -> None:
        if not hasattr(self, "optimal_trajectory"):
            print("No optimal trajectory results to save. Run solve() first.")
            return

        os.makedirs(output_dir, exist_ok=True)

        traj = self.optimal_trajectory

        results_dict = {
            "condition_number": float(traj["condition_number"]),
            "waypoints": traj["waypoints"].tolist(),
            "trajectory_duration": self.trajectory_duration,
            "n_waypoints": self.n_waypoints,
            "sampling_time": self.dt,
        }

        with open(os.path.join(output_dir, "ur10_optimal_trajectory.yaml"), "w") as f:
            yaml.dump(results_dict, f, default_flow_style=False)

        time_vector = np.arange(len(traj["q"])) * self.dt
        df = pd.DataFrame(
            {
                "time": time_vector,
                **{f"q{i+1}": traj["q"][:, i] for i in range(self.model.nq)},
                **{f"dq{i+1}": traj["dq"][:, i] for i in range(self.model.nv)},
                **{f"ddq{i+1}": traj["ddq"][:, i] for i in range(self.model.nv)},
            }
        )
        df.to_csv(os.path.join(output_dir, "ur10_optimal_trajectory.csv"), index=False)

        print(f"Results saved to {output_dir}/ur10_optimal_trajectory.yaml and .csv")


class OptimalTrajectoryIPOPT(BaseOptimalTrajectory):

    def __init__(
        self,
        robot: Any,
        active_joints: List[str],
        config_file: str = "config/ur10_unified_config.yaml",
    ) -> None:
        super().__init__(robot, active_joints, config_file)
        self.logger.info("UR10 OptimalTrajectoryIPOPT initialized")

    def create_ipopt_problem(
        self,
        n_joints: int,
        n_wps: int,
        Ns: int,
        tps: float,
        vel_wps: float,
        acc_wps: float,
        wp_init: np.ndarray,
        vel_wp_init: np.ndarray,
        acc_wp_init: np.ndarray,
        W_stack: np.ndarray,
    ) -> UR10TrajectoryIPOPTProblem:
        return UR10TrajectoryIPOPTProblem(
            self,
            n_joints,
            n_wps,
            Ns,
            tps,
            vel_wps,
            acc_wps,
            wp_init,
            vel_wp_init,
            acc_wp_init,
            W_stack,
        )


class UR10TrajectoryIPOPTProblem(BaseTrajectoryIPOPTProblem):

    def __init__(
        self,
        opt_traj: OptimalTrajectoryIPOPT,
        n_joints: int,
        n_wps: int,
        Ns: int,
        tps: float,
        vel_wps: float,
        acc_wps: float,
        wp_init: np.ndarray,
        vel_wp_init: np.ndarray,
        acc_wp_init: np.ndarray,
        W_stack: np.ndarray,
    ) -> None:
        super().__init__(
            opt_traj,
            n_joints,
            n_wps,
            Ns,
            tps,
            vel_wps,
            acc_wps,
            wp_init,
            vel_wp_init,
            acc_wp_init,
            W_stack,
            "UR10TrajectoryOptimization",
        )
