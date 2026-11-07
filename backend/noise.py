"""Crop Trust Engine - Physical Measurement-Noise Generation Module.

Role 2: Trust Engine Engineer
Approved primary noise model implementation for:
  - Stability analysis and flip-rate estimation (E8)
  - Calibration robustness and noisy-test accuracy (E4 / Role 1)
  - Microclimate and soil measurement sensitivity curves

Reads approved standard deviations (1-sigma) and physical clipping bounds
from backend/models/noise_model.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Canonical feature ordering used across the project
FEATURE_NAMES: Tuple[str, ...] = (
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
)

# Approved physical API clipping bounds: [min, max]
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "N": (0.0, 200.0),
    "P": (0.0, 200.0),
    "K": (0.0, 300.0),
    "temperature": (-10.0, 60.0),
    "humidity": (0.0, 100.0),
    "ph": (0.0, 14.0),
    "rainfall": (0.0, 1000.0),
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "models" / "noise_model.json"


def load_noise_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and parse the approved noise model JSON configuration.

    Args:
        config_path: Optional path to noise_model.json. If None, uses default location.

    Returns:
        Dictionary containing the parsed noise model configuration.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If the configuration is missing required profiles or schema fields.
    """
    resolved_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Noise model configuration file not found at: {resolved_path}"
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "profiles" not in config or "default_profile" not in config:
        raise ValueError(
            f"Invalid noise configuration format in {resolved_path}: "
            "missing 'profiles' or 'default_profile' key."
        )

    return config


def compute_feature_sigmas(
    data: np.ndarray,
    profile_config: Dict[str, Any],
) -> np.ndarray:
    """Compute the 1-sigma standard deviation array matching data shape.

    Formulas approved by Role 1 (Abhishek):
      - Relative features (N, P, K, rainfall):
        sigma = max(relative_ratio * abs(value), sigma_floor)
      - Absolute features (temperature, humidity, ph):
        sigma = fixed approved value

    Args:
        data: NumPy array of shape (7,) or (N, 7) in canonical feature order.
        profile_config: Feature configuration dictionary for the selected profile.

    Returns:
        NumPy array of standard deviations matching data shape.
    """
    features_cfg = profile_config["features"]
    sigmas = np.zeros_like(data, dtype=np.float64)

    # 1D array of 7 features
    if data.ndim == 1:
        if data.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected 1D array of length {len(FEATURE_NAMES)}, got shape {data.shape}"
            )
        for idx, feat in enumerate(FEATURE_NAMES):
            cfg = features_cfg[feat]
            if cfg["scale_type"] == "relative":
                ratio = cfg["relative_ratio"]
                floor = cfg["sigma_floor"]
                sigmas[idx] = max(ratio * abs(data[idx]), floor)
            else:
                sigmas[idx] = cfg["sigma"]
        return sigmas

    # 2D array of (N, 7)
    if data.ndim == 2:
        if data.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected 2D array with {len(FEATURE_NAMES)} columns, got shape {data.shape}"
            )
        for idx, feat in enumerate(FEATURE_NAMES):
            cfg = features_cfg[feat]
            if cfg["scale_type"] == "relative":
                ratio = cfg["relative_ratio"]
                floor = cfg["sigma_floor"]
                # Vectorized element-wise maximum over absolute values
                sigmas[:, idx] = np.maximum(ratio * np.abs(data[:, idx]), floor)
            else:
                sigmas[:, idx] = cfg["sigma"]
        return sigmas

    raise ValueError(f"Data must be 1D or 2D array, got ndim={data.ndim}")


def apply_noise(
    data: Union[np.ndarray, Dict[str, float], List[float]],
    n_samples: int = 1,
    seed: Optional[int] = 42,
    profile: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    return_clip_stats: bool = False,
) -> Union[
    np.ndarray,
    Dict[str, float],
    Tuple[Union[np.ndarray, Dict[str, float]], Dict[str, Any]],
]:
    """Apply approved physical measurement noise to crop feature data.

    Perturbations are zero-mean independent Gaussian perturbations:
      x_noisy = x + N(0, sigma^2)
      x_clipped = clip(x_noisy, min_bound, max_bound)

    Features are perturbed independently in raw dataset units. The original input
    is never mutated.

    Args:
        data: Input data as:
              - 1D array or list of 7 floats: (N, P, K, temp, humidity, ph, rain)
              - 2D array of shape (N, 7)
              - Dict with 7 feature keys
        n_samples: Number of noisy copies to generate per row.
                   For single 1D input with n_samples > 1, returns (n_samples, 7).
                   For 2D input with n_samples == 1, returns (N, 7).
                   For 2D input with n_samples > 1, returns (n_samples, N, 7).
        seed: Random seed for reproducibility (default: 42).
              If None, non-deterministic random state is used.
        profile: Noise profile to use ('primary_1sigma' by default).
        config_path: Optional path to noise_model.json.
        return_clip_stats: If True, returns a tuple (result, clip_stats) recording
                           the count and rate of values that hit physical bounds.

    Returns:
        Noisy data matching input type (or NumPy array), or tuple of (noisy_data, clip_stats).
    """
    config = load_noise_config(config_path)
    selected_profile = profile if profile is not None else config["default_profile"]

    if selected_profile not in config["profiles"]:
        raise ValueError(
            f"Unknown noise profile '{selected_profile}'. "
            f"Available profiles: {list(config['profiles'].keys())}"
        )

    profile_cfg = config["profiles"][selected_profile]
    rng = np.random.default_rng(seed)

    # Track if input was a dict
    is_dict_input = isinstance(data, dict)
    if is_dict_input:
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            raise ValueError(f"Missing required feature(s) in dict input: {missing}")
        arr_input = np.array([float(data[f]) for f in FEATURE_NAMES], dtype=np.float64)
    else:
        arr_input = np.array(data, dtype=np.float64, copy=True)

    lower_bounds = np.array([PHYSICAL_BOUNDS[f][0] for f in FEATURE_NAMES], dtype=np.float64)
    upper_bounds = np.array([PHYSICAL_BOUNDS[f][1] for f in FEATURE_NAMES], dtype=np.float64)

    # Case 1: 1D single input row
    if arr_input.ndim == 1:
        if arr_input.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected 1D input with {len(FEATURE_NAMES)} features, got {arr_input.shape[0]}"
            )

        sigmas = compute_feature_sigmas(arr_input, profile_cfg)

        if n_samples == 1:
            noise = rng.normal(0.0, sigmas, size=len(FEATURE_NAMES))
            unclipped = arr_input + noise
            clipped = np.clip(unclipped, lower_bounds, upper_bounds)
        else:
            # Broadcast sigmas across n_samples: shape (n_samples, 7)
            noise = rng.normal(0.0, sigmas, size=(n_samples, len(FEATURE_NAMES)))
            unclipped = arr_input + noise
            clipped = np.clip(unclipped, lower_bounds, upper_bounds)

    # Case 2: 2D batch of shape (N, 7)
    elif arr_input.ndim == 2:
        if arr_input.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected 2D input with {len(FEATURE_NAMES)} columns, got shape {arr_input.shape}"
            )

        sigmas = compute_feature_sigmas(arr_input, profile_cfg)

        if n_samples == 1:
            noise = rng.normal(0.0, sigmas, size=arr_input.shape)
            unclipped = arr_input + noise
            clipped = np.clip(unclipped, lower_bounds, upper_bounds)
        else:
            # Generate 3D batch: shape (n_samples, N, 7)
            noise = rng.normal(
                0.0,
                np.broadcast_to(sigmas, (n_samples, arr_input.shape[0], arr_input.shape[1])),
            )
            unclipped = arr_input + noise
            clipped = np.clip(unclipped, lower_bounds, upper_bounds)

    else:
        raise ValueError(f"Input data must be 1D or 2D, got ndim={arr_input.ndim}")

    # Compute clipping metrics
    if return_clip_stats:
        clipped_mask = (unclipped < lower_bounds) | (unclipped > upper_bounds)
        total_clipped = int(np.sum(clipped_mask))
        total_elements = int(unclipped.size)
        clip_rate = float(total_clipped / total_elements) if total_elements > 0 else 0.0

        # Per-feature clipping count
        per_feature_clipped: Dict[str, int] = {}
        for idx, feat in enumerate(FEATURE_NAMES):
            # Mask indexing on last axis
            feat_mask = clipped_mask[..., idx]
            per_feature_clipped[feat] = int(np.sum(feat_mask))

        clip_stats: Dict[str, Any] = {
            "total_clipped": total_clipped,
            "total_elements": total_elements,
            "clip_rate": round(clip_rate, 4),
            "per_feature_clipped": per_feature_clipped,
        }

    # Format return value matching input type when applicable
    if is_dict_input and n_samples == 1:
        res: Union[np.ndarray, Dict[str, float]] = {
            FEATURE_NAMES[i]: float(clipped[i]) for i in range(len(FEATURE_NAMES))
        }
    else:
        res = clipped

    if return_clip_stats:
        return res, clip_stats
    return res
