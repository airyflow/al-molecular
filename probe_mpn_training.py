#!/usr/bin/env python3
"""Where does MPN training time go? Times, on N molecules of a dataset:

  1. featurize   SMILES -> RDKit mol -> MolGraph in one process (ms/molecule)
  2. loader      one full pass of MoleculeDataLoader with no model attached:
                 worker start-up plus graph building in the workers, which is
                 the per-epoch featurization cost training pays (graphs are
                 rebuilt every epoch; see MoleculeDataset.batch_graph)
  3. train       MPNN training for --epochs epochs, end to end

at each --batch-sizes value. If (2) per epoch is close to (3) per epoch,
featurization dominates and batch size cannot help much -- which is what
D4's batch-size-1024 run suggested (report sect. 16).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-file", required=True)
    ap.add_argument("--scores-file", required=True, help="row-aligned with --smiles-file (csv/csv.gz with a score column)")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ncpu", type=int, default=8)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[50, 1024])
    args = ap.parse_args()

    from rdkit import Chem
    from molpal.models import model as build_model
    from molpal.models.chemprop.data.data import MoleculeDataLoader, MoleculeDataset, MoleculeDatapoint
    from molpal.models.chemprop.features import MolGraph

    with open(args.smiles_file) as f:
        smis = [next(f).rstrip("\n") for _ in range(args.n)]
    df = pd.read_csv(args.scores_file, nrows=args.n)
    score_col = next(c for c in df.columns if "score" in c.lower())
    ys = -pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=np.float32)
    ok = ~np.isnan(ys)
    smis = [s for s, k in zip(smis, ok) if k]
    ys = ys[ok]
    n = len(smis)
    print(f"[data] {n:,} scored molecules, ncpu={args.ncpu}", flush=True)

    m = min(n, 20_000)
    t = time.perf_counter()
    mols = [Chem.MolFromSmiles(s) for s in smis[:m]]
    t_mol = time.perf_counter() - t
    t = time.perf_counter()
    for mol in mols:
        MolGraph(mol)
    t_graph = time.perf_counter() - t
    print(f"[featurize] {m:,} molecules, one process: SMILES->mol {1e3*t_mol/m:.3f} ms/mol, "
          f"mol->MolGraph {1e3*t_graph/m:.3f} ms/mol", flush=True)

    for bs in args.batch_sizes:
        ds = MoleculeDataset([MoleculeDatapoint([s], [y]) for s, y in zip(smis, ys)])
        loader = MoleculeDataLoader(ds, bs, args.ncpu, pin_memory=False, shuffle=False)
        t = time.perf_counter()
        it = iter(loader)
        next(it)
        t_first = time.perf_counter() - t
        k = 1
        for _ in it:
            k += 1
        t_load = time.perf_counter() - t
        print(f"[loader] bs={bs}: {k:,} batches, first batch after {t_first:.1f}s, full pass {t_load:.1f}s "
              f"= {n / t_load:,.0f} mol/s", flush=True)

        mdl = build_model(model="mpn", conf_method="mve", input_size=2048, test_batch_size=4096,
                          ncpu=args.ncpu, batch_size=bs, epochs=args.epochs)
        t = time.perf_counter()
        mdl.train(smis, ys, retrain=True)
        t_train = time.perf_counter() - t
        print(f"[train] bs={bs}: {args.epochs} epochs in {t_train:.1f}s = {t_train / args.epochs:.1f}s/epoch "
              f"= {n * args.epochs / t_train:,.0f} molecule-epochs/s; data-loader pass alone = "
              f"{t_load:.1f}s ({100 * t_load / (t_train / args.epochs):.0f}% of an epoch)", flush=True)


if __name__ == "__main__":
    main()
