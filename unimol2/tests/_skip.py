"""Shared skip helper for tests that need the (4.2GB, not committed to the
repo) checkpoint file on disk. Uses unittest.SkipTest so these tests skip
cleanly under pytest too, if pytest is ever added to this repo -- pytest
recognizes unittest.SkipTest as a skip, not a failure."""
from __future__ import annotations

import unittest
from pathlib import Path

from unimol2.config import UniMol2Config


def require_checkpoint() -> None:
    checkpoint_path = UniMol2Config().checkpoint_path
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise unittest.SkipTest(
            f"checkpoint not found at {checkpoint_path} -- "
            f"download it first (see models/unimol2/1.1B/download.log for the source URL)"
        )
