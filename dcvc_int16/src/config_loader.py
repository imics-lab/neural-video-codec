"""Shared YAML configuration loader for all DCVC scripts.

Priority order (highest wins):
  1. CLI argument (explicit --flag on the command line)
  2. config/config.yaml value  (project-local user config)
  3. argparse default   (hardcoded fallback in each script)

Usage in any script::

    from src.config_loader import load_config, apply_config_defaults

    def parse_args():
        cfg = load_config()
        parser = argparse.ArgumentParser(...)
        parser.add_argument("--bundle_path", default="models/...")
        # ... define all args ...
        apply_config_defaults(parser, cfg, "encode")
        return parser.parse_args()

The `section` argument in `apply_config_defaults` maps directly to a top-level
YAML key in config/config.yaml. Model paths are always pulled from the ``models``
section regardless of which section is active.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_UNSET = object()


def load_config(path: str | Path | None = None) -> dict:
    """Load and return the YAML config as a plain dict.

    Searches for the config file in this order:
    1. ``path`` argument (if provided)
    2. ``DCVC_CONFIG`` environment variable
    3. ``<project-root>/config/config.yaml`` (preferred)
    4. ``<project-root>/config.yaml`` (legacy fallback)

    Returns an empty dict if no config file is found (not an error).

    Args:
        path: Explicit path to a YAML config file.

    Returns:
        Parsed YAML as a nested dict.
    """
    import yaml

    if path is None:
        env_path = os.environ.get("DCVC_CONFIG")
        if env_path:
            path = Path(env_path)
        else:
            preferred = ROOT / "config" / "config.yaml"
            legacy = ROOT / "config.yaml"
            path = preferred if preferred.exists() else legacy

    path = Path(path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)

    return data if isinstance(data, dict) else {}


def get(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safely navigate a nested dict with a chain of keys.

    Example::

        qp = get(cfg, "encode", "qp_p", default=32)

    Args:
        cfg: Config dict returned by :func:`load_config`.
        *keys: Sequence of keys to traverse.
        default: Value returned when any key is missing.

    Returns:
        The value at the nested path, or *default*.
    """
    node = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, _UNSET)
        if node is _UNSET:
            return default
    return node


def apply_config_defaults(
    parser: "argparse.ArgumentParser",
    cfg: dict,
    section: str,
) -> None:
    """Override argparse defaults with values from *cfg[section]* and *cfg[models]*.

    Only keys that exist in *cfg* are applied — missing keys leave the argparse
    default untouched. This ensures CLI args always take precedence.

    The following automatic cross-section mappings are applied so every script
    gets model paths without repeating them in each section:

    - ``cfg.models.bundle``          → ``bundle_path``
    - ``cfg.models.checkpoint_i``    → ``model_path_i``
    - ``cfg.models.checkpoint_p``    → ``model_path_p``
    - ``cfg.models.frozen_entropy``  → ``frozen_entropy_path``

    Args:
        parser: The argparse parser to update.
        cfg: Config dict from :func:`load_config`.
        section: Top-level section key in *cfg* (e.g. ``"encode"``).
    """
    overrides: dict[str, Any] = {}

    # Cross-section model path shortcuts available to every script
    _map_if_present(cfg, overrides, ("models", "bundle"),         "bundle_path")
    _map_if_present(cfg, overrides, ("models", "checkpoint_i"),   "model_path_i")
    _map_if_present(cfg, overrides, ("models", "checkpoint_p"),   "model_path_p")
    _map_if_present(cfg, overrides, ("models", "frozen_entropy"), "frozen_entropy_path")

    # Section-specific values
    section_data = get(cfg, section)
    if isinstance(section_data, dict):
        for key, value in section_data.items():
            if value is not None:
                overrides[key] = value

    if overrides:
        parser.set_defaults(**overrides)


def add_config_arg(parser: "argparse.ArgumentParser") -> None:
    """Add a ``--config`` argument to *parser*.

    Call this before :func:`apply_config_defaults` so users can point to a
    non-default config file on the command line.

    Example workflow in a script::

        parser = argparse.ArgumentParser(...)
        add_config_arg(parser)
        known, _ = parser.parse_known_args()
        cfg = load_config(known.config)
        apply_config_defaults(parser, cfg, "encode")
        args = parser.parse_args()
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML config file. "
            "Defaults to config/config.yaml in the project root, "
            "or the DCVC_CONFIG environment variable."
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_if_present(
    cfg: dict,
    overrides: dict,
    src_keys: tuple[str, ...],
    dest_key: str,
) -> None:
    """Write cfg[src_keys] → overrides[dest_key] only if the value is set."""
    value = get(cfg, *src_keys)
    if value is not None:
        overrides[dest_key] = value
