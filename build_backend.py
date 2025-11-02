"""Wrapper build backend that keeps editable installs working on older frontends.

Some pip 22.x distributions probe the build backend for PEP 660 support from the
invoking (often system) Python environment, which may bundle an old
``setuptools`` that predates ``build_editable``.  Defining the hooks here means
that probe succeeds, while the actual work is delegated to the up-to-date
``setuptools`` that pip installs in the isolated build environment.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional

import sys


_EDITABLE_HOOKS = {
    "build_editable",
    "prepare_metadata_for_build_editable",
    "get_requires_for_build_editable",
}


def _prioritize_build_env_on_sys_path() -> None:
    build_env_paths = [p for p in sys.path if "pip-build-env-" in p]
    if not build_env_paths:
        return
    for path in build_env_paths:
        try:
            sys.path.remove(path)
        except ValueError:
            continue
    for path in reversed(build_env_paths):
        sys.path.insert(0, path)


def _call_backend(method: str, *args: Any, **kwargs: Any) -> Any:
    _prioritize_build_env_on_sys_path()
    backend = importlib.import_module("setuptools.build_meta")
    try:
        hook = getattr(backend, method)
    except AttributeError as exc:
        if method in _EDITABLE_HOOKS:
            import setuptools

            raise NotImplementedError(
                "Editable installs require setuptools>=67. "
                f"Current setuptools version is {setuptools.__version__}."
            ) from exc
        raise
    return hook(*args, **kwargs)


def build_wheel(
    wheel_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    return _call_backend(
        "build_wheel",
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


def build_sdist(sdist_directory: str, config_settings: Optional[Dict[str, Any]] = None) -> str:
    return _call_backend("build_sdist", sdist_directory, config_settings=config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: Optional[Dict[str, Any]] = None
) -> str:
    return _call_backend(
        "prepare_metadata_for_build_wheel",
        metadata_directory,
        config_settings=config_settings,
    )


def get_requires_for_build_wheel(
    config_settings: Optional[Dict[str, Any]] = None
) -> Iterable[str]:
    return _call_backend("get_requires_for_build_wheel", config_settings=config_settings)


def get_requires_for_build_sdist(
    config_settings: Optional[Dict[str, Any]] = None
) -> Iterable[str]:
    return _call_backend("get_requires_for_build_sdist", config_settings=config_settings)


def build_editable(
    wheel_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    return _call_backend(
        "build_editable",
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: Optional[Dict[str, Any]] = None
) -> str:
    return _call_backend(
        "prepare_metadata_for_build_editable",
        metadata_directory,
        config_settings=config_settings,
    )


def get_requires_for_build_editable(
    config_settings: Optional[Dict[str, Any]] = None
) -> Iterable[str]:
    return _call_backend(
        "get_requires_for_build_editable",
        config_settings=config_settings,
    )
