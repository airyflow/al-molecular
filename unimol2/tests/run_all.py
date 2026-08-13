#!/usr/bin/env python3
"""Runs every test in unimol2/tests/. Usage: python3 -m unimol2.tests.run_all
(run from the al-molecular repo root, so `unimol2` is importable)."""
from __future__ import annotations

from . import test_checkpoint_loading, test_feature_pipeline, test_model_invariants

if __name__ == "__main__":
    for module in (test_feature_pipeline, test_checkpoint_loading, test_model_invariants):
        print(f"\n=== {module.__name__} ===")
        module._run_all()
