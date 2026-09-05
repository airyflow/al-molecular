"""Verifies the ported model's architecture exactly matches the real
checkpoint's structure. Requires the checkpoint file on disk; skips (not
fails) if it's absent.

Run directly: `python3 -m unimol1.tests.test_checkpoint_loading`
"""
from __future__ import annotations

from ._skip import require_checkpoint


def test_checkpoint_loads_with_only_hidden_layer_missing() -> None:
    require_checkpoint()
    from unimol1 import build_model_from_checkpoint

    # build_model_from_checkpoint() itself raises RuntimeError on any
    # missing key outside hidden_layer.*, or any unexpected key at all --
    # reaching this line without an exception IS the assertion.
    model = build_model_from_checkpoint()

    assert len(model.encoder.layers) == 15, len(model.encoder.layers)
    # 31, not len(UNIMOL_DICT)=30 -- see load_production_dictionary()'s
    # docstring on the appended [MASK] symbol the checkpoint expects.
    assert model.embed_tokens.weight.shape == (31, 512), model.embed_tokens.weight.shape


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
