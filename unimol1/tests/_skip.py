"""Shared skip helper for tests that need the real Uni-Mol checkpoint file
on disk. Uses unittest.SkipTest so these tests skip cleanly under pytest
too (recognized as a skip, not a failure), matching unimol2/tests/_skip.py."""
from __future__ import annotations

import unittest
from pathlib import Path

from unimol1.config import UniMolConfig


def require_checkpoint() -> None:
    checkpoint_path = UniMolConfig().checkpoint_path
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise unittest.SkipTest(f"checkpoint not found at {checkpoint_path}")
