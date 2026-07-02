import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, List, Optional, Tuple, Union
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from .error_handling import DataProcessingError, validate_input_data

logger = logging.getLogger(__name__)


class DataProcessor:

    _cache = {}
    _cache_size_limit = 100

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
        logger.info("DataProcessor cache cleared")

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        return {"cache_size": len(cls._cache), "cache_limit": cls._cache_size_limit}

    @staticmethod
    @lru_cache(maxsize=128)
    def _cached_filter_design(
        filter_type: str, cutoff: float, fs: float, order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if filter_type == "butterworth":
            return signal.butter(order, cutoff, fs=fs, btype="low")
        elif filter_type == "chebyshev":
            return signal.cheby1(order, 0.5, cutoff, fs=fs, btype="low")
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    @staticmethod
    @validate_input_data
    def vectorized_differentiation(
        data: np.ndarray, dt: float = 0.01, method: str = "gradient"
    ) -> Tuple[np.ndarray, np.ndarray]:
        if method == "gradient":
            velocities = np.gradient(data, dt, axis=0)
            accelerations = np.gradient(velocities, dt, axis=0)
        elif method == "finite_diff":
            velocities = np.diff(data, axis=0) / dt
            accelerations = np.diff(velocities, axis=0) / dt
            velocities = np.vstack([velocities, velocities[-1:]])
            accelerations = np.vstack([accelerations, accelerations[-1:]])
        else:
            raise ValueError(f"Unknown method: {method}")

        return velocities, accelerations

    @staticmethod
    @validate_input_data
    def parallel_filtering(
        data_list: List[np.ndarray],
        cutoff: float = 10.0,
        fs: float = 100.0,
        filter_type: str = "butterworth",
        order: int = 4,
        max_workers: Optional[int] = None,
    ) -> List[np.ndarray]:
        b, a = DataProcessor._cached_filter_design(filter_type, cutoff, fs, order)

        def filter_single_array(data):
            if data.ndim == 1:
                return signal.filtfilt(b, a, data)
            else:
                filtered = np.zeros_like(data)
                for i in range(data.shape[1]):
                    filtered[:, i] = signal.filtfilt(b, a, data[:, i])
                return filtered

        if len(data_list) > 1 and data_list[0].size > 1000:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(filter_single_array, data): i
                    for i, data in enumerate(data_list)
                }

                filtered_data = [None] * len(data_list)
                for future in as_completed(futures):
                    idx = futures[future]
                    filtered_data[idx] = future.result()

                return filtered_data
        else:
            return [filter_single_array(data) for data in data_list]

    @staticmethod
    @validate_input_data
    def load_csv_data(
        file_paths: Union[str, List[str]], validate_shapes: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        try:
            if isinstance(file_paths, str):
                data = pd.read_csv(file_paths).to_numpy()
                return data
            else:
                data_arrays = []
                expected_shape = None

                for i, file_path in enumerate(file_paths):
                    data = pd.read_csv(file_path).to_numpy()
                    data_arrays.append(data)

                    if validate_shapes:
                        if expected_shape is None:
                            expected_shape = data.shape
                        elif data.shape != expected_shape:
                            raise DataProcessingError(
                                f"Shape mismatch in file {i}: expected "
                                f"{expected_shape}, got {data.shape}"
                            )

                return data_arrays

        except Exception as e:
            raise DataProcessingError(f"Failed to load CSV data: {e}") from e

    @staticmethod
    @lru_cache(maxsize=128)
    def create_filter_coefficients(
        filter_type: str, cutoff_freq: float, sampling_freq: float, order: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        nyquist = sampling_freq / 2

        if filter_type == "lowpass":
            normalized_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(order, normalized_cutoff, btype="low")
        elif filter_type == "highpass":
            normalized_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(order, normalized_cutoff, btype="high")
        elif filter_type == "bandpass":
            if not isinstance(cutoff_freq, (list, tuple)) or len(cutoff_freq) != 2:
                raise ValueError("Bandpass filter requires two cutoff frequencies")
            normalized_cutoff = [f / nyquist for f in cutoff_freq]
            b, a = signal.butter(order, normalized_cutoff, btype="band")
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        return b, a

    @staticmethod
    @validate_input_data
    def apply_filter(
        data: np.ndarray,
        filter_type: str,
        cutoff_freq: float,
        sampling_freq: float,
        order: int = 4,
        axis: int = 0,
    ) -> np.ndarray:
        try:
            b, a = DataProcessor.create_filter_coefficients(
                filter_type, cutoff_freq, sampling_freq, order
            )

            filtered_data = signal.filtfilt(b, a, data, axis=axis)

            return filtered_data

        except Exception as e:
            raise DataProcessingError(f"Filter application failed: {e}") from e

    @staticmethod
    @validate_input_data
    def compute_derivatives(
        positions: np.ndarray,
        timestamps: np.ndarray,
        method: str = "central",
        filter_params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if positions.shape[0] != timestamps.shape[0]:
            raise DataProcessingError(
                f"Position and timestamp shapes don't match: "
                f"{positions.shape[0]} vs {timestamps.shape[0]}"
            )

        n_samples, n_joints = positions.shape

        dt = np.diff(timestamps)
        if np.any(dt <= 0):
            raise DataProcessingError("Timestamps must be strictly increasing")

        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)

        if method == "central":
            for i in range(1, n_samples - 1):
                dt_prev = timestamps[i] - timestamps[i - 1]
                dt_next = timestamps[i + 1] - timestamps[i]
                dt_avg = (dt_prev + dt_next) / 2

                velocities[i, :] = (positions[i + 1, :] - positions[i - 1, :]) / (
                    2 * dt_avg
                )

            velocities[0, :] = (positions[1, :] - positions[0, :]) / dt[0]
            velocities[-1, :] = (positions[-1, :] - positions[-2, :]) / dt[-1]

        elif method == "forward":
            for i in range(n_samples - 1):
                velocities[i, :] = (positions[i + 1, :] - positions[i, :]) / dt[i]
            velocities[-1, :] = velocities[-2, :]

        elif method == "backward":
            velocities[0, :] = velocities[1, :]
            for i in range(1, n_samples):
                velocities[i, :] = (positions[i, :] - positions[i - 1, :]) / dt[i - 1]
        else:
            raise ValueError(f"Unknown differentiation method: {method}")

        dt_vel = np.diff(timestamps)
        for i in range(1, n_samples - 1):
            dt_avg = (dt_vel[i - 1] + dt_vel[i]) / 2
            accelerations[i, :] = (velocities[i + 1, :] - velocities[i - 1, :]) / (
                2 * dt_avg
            )

        accelerations[0, :] = (velocities[1, :] - velocities[0, :]) / dt_vel[0]
        accelerations[-1, :] = (velocities[-1, :] - velocities[-2, :]) / dt_vel[-1]

        if filter_params:
            sampling_freq = 1.0 / np.mean(dt)
            velocities = DataProcessor.apply_filter(
                velocities, sampling_freq=sampling_freq, **filter_params
            )
            accelerations = DataProcessor.apply_filter(
                accelerations, sampling_freq=sampling_freq, **filter_params
            )

        return velocities, accelerations

    @staticmethod
    @validate_input_data
    def decimate_data(
        data_dict: Dict[str, np.ndarray], factor: int, method: str = "uniform"
    ) -> Dict[str, np.ndarray]:
        if factor <= 1:
            return data_dict

        decimated_data = {}

        for key, data in data_dict.items():
            if data is None:
                decimated_data[key] = None
                continue

            if method == "uniform":
                decimated_data[key] = data[::factor]
            elif method == "adaptive":
                decimated_data[key] = data[::factor]
            else:
                raise ValueError(f"Unknown decimation method: {method}")

        return decimated_data

    @staticmethod
    def validate_data_consistency(
        data_dict: Dict[str, np.ndarray], expected_keys: Optional[List[str]] = None
    ) -> None:
        if expected_keys:
            missing_keys = set(expected_keys) - set(data_dict.keys())
            if missing_keys:
                raise DataProcessingError(f"Missing required keys: {missing_keys}")

        reference_shape = None
        reference_key = None

        for key, data in data_dict.items():
            if data is not None:
                reference_shape = data.shape
                reference_key = key
                break

        if reference_shape is None:
            raise DataProcessingError("No valid data arrays found")

        for key, data in data_dict.items():
            if data is not None and data.shape != reference_shape:
                raise DataProcessingError(
                    f"Shape mismatch: {reference_key} has shape {reference_shape}, "
                    f"but {key} has shape {data.shape}"
                )

    @staticmethod
    @validate_input_data
    def remove_outliers(
        data: np.ndarray, method: str = "iqr", threshold: float = 1.5, axis: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        if method == "iqr":
            q1 = np.percentile(data, 25, axis=axis, keepdims=True)
            q3 = np.percentile(data, 75, axis=axis, keepdims=True)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outlier_mask = (data < lower_bound) | (data > upper_bound)

        elif method == "zscore":
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            z_scores = np.abs((data - mean) / (std + 1e-10))

            outlier_mask = z_scores > threshold

        elif method == "modified_zscore":
            median = np.median(data, axis=axis, keepdims=True)
            mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
            modified_z_scores = 0.6745 * (data - median) / (mad + 1e-10)

            outlier_mask = np.abs(modified_z_scores) > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        cleaned_data = data.copy()
        cleaned_data[outlier_mask] = np.nan

        return cleaned_data, outlier_mask

    @staticmethod
    @lru_cache(maxsize=32)
    def compute_performance_metrics(
        data_signature: str, n_samples: int, n_joints: int
    ) -> Dict[str, float]:
        return {
            "complexity_score": n_samples * n_joints / 1000.0,
            "recommended_batch_size": min(1000, max(100, n_samples // 10)),
            "parallel_threshold": 500 * n_joints,
            "cache_effective": n_samples > 100,
        }

    @classmethod
    def create_processing_pipeline(cls, steps: List[str], **kwargs) -> List[callable]:
        pipeline = []

        for step in steps:
            if step == "filter":
                cutoff = kwargs.get("cutoff", 10.0)
                fs = kwargs.get("fs", 100.0)
                pipeline.append(
                    lambda data: cls.apply_filter(
                        data, "lowpass", cutoff_freq=cutoff, sampling_freq=fs
                    )
                )
            elif step == "differentiate":
                dt = kwargs.get("dt", 0.01)
                pipeline.append(
                    lambda data: cls.vectorized_differentiation(data, dt)[0]
                )
            elif step == "outliers":
                method = kwargs.get("outlier_method", "iqr")
                threshold = kwargs.get("outlier_threshold", 1.5)
                pipeline.append(
                    lambda data: cls.remove_outliers(
                        data, method=method, threshold=threshold
                    )[0]
                )
            elif step == "decimate":
                factor = kwargs.get("decimation_factor", 2)
                pipeline.append(lambda data: cls.decimate_data(data, factor))

        return pipeline


def load_trajectory_data_csv(
    position_file: str, torque_file: str, validate: bool = True
) -> Dict[str, np.ndarray]:
    processor = DataProcessor()

    positions, torques = processor.load_csv_data(
        [position_file, torque_file], validate_shapes=validate
    )

    return {
        "positions": positions,
        "torques": torques,
        "velocities": None,
        "accelerations": None,
        "timestamps": None,
    }


def apply_butterworth_filter(
    data: np.ndarray, cutoff_freq: float, sampling_freq: float, order: int = 4
) -> np.ndarray:
    return DataProcessor.apply_filter(
        data, "lowpass", cutoff_freq, sampling_freq, order
    )
