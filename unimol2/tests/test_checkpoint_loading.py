"""Verifies the ported model's architecture exactly matches the real
checkpoint's structure -- the load-bearing gate from this port's build
process (see unimol2/checkpoint.py's docstring). Requires the checkpoint
file on disk; skips (not fails) if it's absent.

Run directly: `python3 -m unimol2.tests.test_checkpoint_loading`
"""
from __future__ import annotations

from ._skip import require_checkpoint


def test_checkpoint_loads_with_zero_missing_or_unexpected_keys() -> None:
    require_checkpoint()
    from unimol2 import build_model_from_checkpoint

    # build_model_from_checkpoint() itself raises RuntimeError on any
    # missing key, or any unexpected key outside the deliberately-unported
    # movement_pred_head/lm_head/classification_heads prefixes (see
    # unimol2/checkpoint.py) -- reaching this line without an exception IS
    # the assertion.
    model = build_model_from_checkpoint()

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 1_119_516_288, f"unexpected param count: {n_params:,}"
    assert len(model.encoder.layers) == 64, len(model.encoder.layers)


def _run_all() -> None:
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    import unittest
    n_skipped = 0
    for t in tests:
        print(f"running {t.__name__} ...", end=" ")
        try:
            t()
        except unittest.SkipTest as e:
            print(f"SKIP ({e})")
            n_skipped += 1
            continue
        print("PASS")
    print(f"\n{len(tests) - n_skipped}/{len(tests)} tests passed ({n_skipped} skipped).")


if __name__ == "__main__":
    _run_all()
