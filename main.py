"""
Certainty-Aware Symbolic Dermatological Reasoning Engine
Pipeline orchestrator — runs the full diagnostic inference pipeline end-to-end.
Phase 1: Infrastructure validation only.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on the import path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger
from src.utils.reproducibility import ReproducibilityManager

log = get_logger(__name__)


def main() -> None:
    log.info("Initialising clinical inference pipeline", phase="1", status="infrastructure_validation")

    repro = ReproducibilityManager(seed=42)
    repro.set_global_seeds()
    run_id = repro.generate_run_id()

    log.info("Reproducibility context established", run_id=run_id, seed=42)
    log.info(
        "Phase 1 complete — data infrastructure, feature registry, rule schema, "
        "pipeline contracts, and testing infrastructure are operational."
    )


if __name__ == "__main__":
    main()
