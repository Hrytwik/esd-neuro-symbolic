"""
Reproducibility management for the Certainty-Aware Symbolic Reasoning Engine.

Ensures deterministic behaviour across runs by seeding all random number
generators and capturing the execution environment for archival.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class ReproducibilityManager:
    """
    Manages global seed state and environment capture for reproducible runs.

    Attributes
    ----------
    seed:
        Master integer seed applied to all random number generators.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._run_id: str | None = None

    # ── Seed management ───────────────────────────────────────────────────────

    def set_global_seeds(self) -> None:
        """Apply the master seed to Python, NumPy, and (if available) PyTorch."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        try:
            import torch  # optional dependency — Phase 3+
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass

        log.debug("Global seeds applied", seed=self.seed)

    # ── Run identity ──────────────────────────────────────────────────────────

    def generate_run_id(self) -> str:
        """
        Generate a deterministic-then-timestamped run identifier.

        The run ID is a 12-character hex prefix derived from the seed value
        combined with a UTC timestamp, providing both traceability and
        uniqueness across repeated executions.
        """
        now = datetime.now(timezone.utc)
        raw = f"seed={self.seed}|ts={now.isoformat()}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        self._run_id = f"run-{digest}-{now.strftime('%Y%m%dT%H%M%S')}"
        log.debug("Run ID generated", run_id=self._run_id)
        return self._run_id

    @property
    def run_id(self) -> str | None:
        return self._run_id

    # ── Environment capture ───────────────────────────────────────────────────

    def capture_environment(self) -> dict[str, Any]:
        """
        Capture a snapshot of the execution environment for archival in
        experiment outputs. Returned dict is JSON-serialisable.
        """
        env: dict[str, Any] = {
            "run_id":      self._run_id,
            "seed":        self.seed,
            "python":      sys.version,
            "platform":    platform.platform(),
            "cpu_count":   os.cpu_count(),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "packages":    self._installed_packages(),
        }
        log.debug("Environment captured", run_id=self._run_id)
        return env

    @staticmethod
    def _installed_packages() -> dict[str, str]:
        """Return a mapping of relevant package names to their versions."""
        relevant = {
            "numpy", "pandas", "pydantic", "structlog",
            "ucimlrepo", "pyyaml", "jsonschema",
            "xgboost", "scikit-learn", "shap", "torch",
        }
        packages: dict[str, str] = {}
        try:
            import importlib.metadata as meta
            for pkg in relevant:
                try:
                    packages[pkg] = meta.version(pkg)
                except meta.PackageNotFoundError:
                    pass
        except ImportError:
            pass
        return packages
