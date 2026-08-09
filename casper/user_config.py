"""User configuration loader.

This module provides a single place to read user-editable configuration.
The primary configuration file is ``casper/user_config.json``.

If the file is missing or incomplete, defaults are used.
Environment variables may override some paths.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Find the repo root by looking for pyproject.toml
current = Path(__file__).resolve().parent
PROJECT_ROOT = current
while current.parent != current:
    if (current / "pyproject.toml").exists():
        PROJECT_ROOT = current
        break
    current = current.parent

# Allow overriding paths via environment variables for portability.
# If an env var is provided but points to a non-existent location, fall back to the repo root.


def _resolve_path(env_var: str, default: Path) -> str:
    """Resolve a path from an env var, with a fallback to `default`.

    If the env var is not set or points to a non-existent location, fall back to
    the repository root. If it is a relative path, resolve it relative to the
    repository root.
    """

    raw = os.getenv(env_var)
    if not raw:
        return str(default)

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = default / candidate

    if candidate.exists():
        return str(candidate)

    print(
        f"WARNING: {env_var}={raw} does not exist. Falling back to project root: {default}",
        file=sys.stderr,
    )
    return str(default)


CASPER_INPUT_PATH = _resolve_path("CASPER_INPUT_PATH", PROJECT_ROOT)
CASPER_OUTPUT_PATH = _resolve_path("CASPER_OUTPUT_PATH", PROJECT_ROOT)

# If the user explicitly set CASPER_INPUT_PATH, we will treat missing files as fatal.
CASPER_INPUT_PATH_SET = os.getenv("CASPER_INPUT_PATH") is not None


def _resolve_input_subpath(*parts: str) -> str:
    """Resolve a path under the configured input dir.

    If CASPER_INPUT_PATH is explicitly set and the expected file/subdir is
    missing, we raise a clear error to force the user to fix the configuration.

    If CASPER_INPUT_PATH is not set, we fall back to the repository defaults.
    """

    candidate = Path(CASPER_INPUT_PATH).joinpath(*parts)
    if candidate.exists():
        return str(candidate)

    if CASPER_INPUT_PATH_SET:
        raise FileNotFoundError(
            f"CASPER_INPUT_PATH is set to '{CASPER_INPUT_PATH}', "
            f"but the required input path does not exist: '{candidate}'.\n"
            "Please set CASPER_INPUT_PATH to a directory that contains the full "
            "CASPER input tree (e.g. 'casper/inputs/params/...'), or unset "
            "CASPER_INPUT_PATH to use the repository defaults."
        )

    # Fall back to repository defaults when CASPER_INPUT_PATH is not explicitly set.
    fallback = Path(PROJECT_ROOT).joinpath(*parts)
    if fallback.exists():
        return str(fallback)

    return str(candidate)


DEFAULTS: Dict[str, Any] = {
    "io_paths": {
        "plot": True,
        "normalize": True,
        "spectra_dir_path": _resolve_input_subpath("casper/inputs/spectra/test_spectra/"),
        "param_path": _resolve_input_subpath("casper/inputs/params/param_file_test.dat"),
        "output_dir_path": "outputs/",
        "output_file_name": "sample",
    },
    "dirs": {
        "output_dir": os.path.join(CASPER_OUTPUT_PATH, "outputs/"),
        "npsave_dir": os.path.join(CASPER_OUTPUT_PATH, "npsave/"),
    },
    "params": {
        "PLOT_LINEWIDTH": 0.25,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts, with `override` taking priority."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _load_user_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load and merge the user config JSON file with the module defaults.

    Parameters
    ----------
    config_path : Path, optional
        Path to the user config JSON file. Defaults to ``user_config.json``
        next to this module. Exposed as a parameter (rather than hardcoded)
        so this function can be unit tested against arbitrary config files.

    Returns
    -------
    dict
        The merged configuration dictionary (``DEFAULTS`` deep-merged with
        the contents of ``config_path``, if present and valid).
    """

    if config_path is None:
        config_path = Path(__file__).resolve().parent / "user_config.json"

    if not config_path.exists():
        return dict(DEFAULTS)

    try:
        with config_path.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
    except Exception:
        return dict(DEFAULTS)

    if not isinstance(user_config, dict):
        return dict(DEFAULTS)

    # Expand environment variables in string values
    def expand_vars(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        elif isinstance(obj, dict):
            return {k: expand_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_vars(item) for item in obj]
        else:
            return obj

    user_config = expand_vars(user_config)

    # If the user config references CASPER_INPUT_PATH but the referenced files
    # or directories don't exist, fall back to the repository defaults.
    def _normalize_input_path(path_str: str) -> str:
        if not isinstance(path_str, str):
            return path_str

        if os.path.exists(path_str):
            return path_str

        # Only apply fallback for paths that begin with the resolved CASPER_INPUT_PATH.
        if path_str.startswith(str(CASPER_INPUT_PATH)):
            rel = os.path.relpath(path_str, start=str(CASPER_INPUT_PATH))
            return _resolve_input_subpath(rel)

        return path_str

    if isinstance(user_config, dict):
        io_paths = user_config.get("io_paths")
        if isinstance(io_paths, dict):
            if "param_path" in io_paths:
                io_paths["param_path"] = _normalize_input_path(io_paths["param_path"])
            if "spectra_dir_path" in io_paths:
                io_paths["spectra_dir_path"] = _normalize_input_path(io_paths["spectra_dir_path"])

    return _deep_merge(DEFAULTS, user_config)


USER_CONFIG: Dict[str, Any] = _load_user_config()

# Allow environment variable overrides for output directories.
# These are useful when running from CI / containers / temporary dirs.
if env_out := os.getenv("CASPER_OUTPUT_DIR"):
    USER_CONFIG.setdefault("dirs", {})["output_dir"] = env_out
if env_np := os.getenv("CASPER_NPSAVE_DIR"):
    USER_CONFIG.setdefault("dirs", {})["npsave_dir"] = env_np


def get(path: str, default: Any = None) -> Any:
    """Get a nested config value using dot-separated keys.

    Examples:
        get("io_paths.plot")
        get("dirs.output_dir")
    """

    parts = path.split(".") if path else []
    current: Any = USER_CONFIG
    for part in parts:
        if not isinstance(current, dict):
            return default
        current = current.get(part, default)
        if current is default:
            break
    return current
