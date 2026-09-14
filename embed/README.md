# Embedding generation

How to generate molecular embeddings for any SMILES pool, with any of the
backbones this repo supports. This document is only about *extraction* --
for the active-learning pipeline that consumes the resulting embeddings
(`run_experiment.py`, `LTAllSurrogate`, etc.), see the repo's top-level
`README.md`.

No Python environment yet? `environment.yml` at the repo root builds one
(`conda env create -f environment.yml`) -- see its own comments for the
one extra manual step (installing PyTorch, which is CUDA-build-specific
so it isn't pinned in the file itself).

## The shared design

Every backbone follows the same shape: a
`embed/compute/compute_<backbone>_embeddings_chunk.py` script takes a
slice ("chunk") of a plain-text SMILES file and writes one independent
`.npy` file for that chunk; `embed/stitch/stitch_embedding_chunks.py`
concatenates all chunks into the final `(N, D)` array
`molpal/featurizer.py`'s `EmbeddingFeaturizer` (and thus
`run_experiment.py`) expects. This works identically whether you have 1
chunk (a quick local test) or 1000 (a real multi-million-molecule pool
split across a SLURM array) -- `--chunk-id 0 --num-chunks 1` is just the
degenerate one-chunk case of the same machinery.

Your input is always a plain text file, one SMILES string per line:
```bash
cat > my_smiles.txt << 'EOF'
CCO
c1ccccc1
CC(=O)Oc1ccccc1C(=O)O
EOF
```

## Quick reference

| Backbone | Dim | Compute script | Conformers needed? |
|---|---|---|---|
| `grover` | 1600 | `compute_grover_embeddings_chunk.py` | no (2D graph only) |
| `grover3400` | 3400 | `compute_grover3400_embeddings_chunk.py` | no (2D graph only) |
| `molformer` | 768 | `compute_molformer_embeddings_chunk.py` | no (tokenized string) |
| `mhgged` | 1024 | `compute_mhgged_embeddings_chunk.py` | no (2D graph grammar) |
| `smited` | 768 | `compute_smited_embeddings_chunk.py` | no (tokenized string) |
| `unimol` (v1) | 512 | `compute_unimol_embeddings_chunk.py` | **yes** -- two-stage |
| `unimol2` | 1536 | `compute_unimol2_embeddings_chunk.py` | **yes** -- one- or two-stage |

`grover` vs `grover3400`: **use `grover3400` for new extraction** unless
you specifically need parity with an older muben-based run. `grover3400`
is the real `tencent-ailab/grover` "both" fingerprint (3400-d), verified
bit-exact against a reference shard. `grover` (1600-d) reproduces an
earlier, simplified muben-based fingerprint (mean-pooling only, 2 of 4
real GROVER readout terms, no RDKit-2D features).

`unimol` vs `unimol2`: different models, not versions of the same
pipeline -- Uni-Mol v1 (47.6M params, 512-d) and Uni-Mol2 (1.12B params,
1536-d) are separate architectures with separate compute scripts. Uni-Mol2
being ~23x larger means its GPU forward pass dominates wall-clock time
regardless of extraction design; see "Conformer-based backbones" below.

## Quick local test (no SLURM, any single backbone)

Run from the repo root, with one chunk covering the whole file:
```bash
python embed/compute/compute_molformer_embeddings_chunk.py \
    --smiles-file my_smiles.txt --chunk-id 0 --num-chunks 1 \
    --chunks-dir /tmp/_molformer_chunks

python embed/compute/compute_smited_embeddings_chunk.py \
    --smiles-file my_smiles.txt --chunk-id 0 --num-chunks 1 \
    --chunks-dir /tmp/_smited_chunks

python embed/compute/compute_mhgged_embeddings_chunk.py \
    --smiles-file my_smiles.txt --chunk-id 0 --num-chunks 1 \
    --chunks-dir /tmp/_mhgged_chunks

python embed/compute/compute_grover3400_embeddings_chunk.py \
    --smiles-file my_smiles.txt --chunk-id 0 --num-chunks 1 \
    --chunks-dir /tmp/_grover3400_chunks
```
Each writes exactly one file: `<chunks-dir>/<backbone>_embeddings_chunk_00000.npy`,
shape `(N, D)` where `N` = your line count. `--total-count` is optional
(it just skips an O(N) line-count scan -- worth passing at real scale, not
needed for a quick test).

## Conformer-based backbones (Uni-Mol v1, Uni-Mol2)

Unlike the backbones above, Uni-Mol v1 and Uni-Mol2 need a 3D conformer
per molecule before the model can run. Both support splitting this into
**Stage 1** (RDKit conformer generation, CPU-only, cheap and
embarrassingly parallel) and **Stage 2** (the actual model forward pass,
GPU-bound) -- so Stage 1 can run on plain CPU nodes instead of competing
for scarce GPU allocation, and Stage 2's GPU time isn't spent waiting on
CPU-bound conformer generation.

**Uni-Mol v1** (two-stage only -- no single-stage mode):
```bash
# Stage 1 -- conformers -> LMDB chunk file(s)
python generate_unimol_conformers_chunk.py \
    --smiles-file my_smiles.txt --out-dir /tmp/_unimol_conformers \
    --chunk-id 0 --num-chunks 1 --n-conformer 1

# Stage 2 -- embeddings, reading Stage 1's output
python embed/compute/compute_unimol_embeddings_chunk.py \
    --smiles-file my_smiles.txt \
    --conformer-chunks-dir /tmp/_unimol_conformers/_chunks \
    --chunk-id 0 --num-chunks 1 --chunks-dir /tmp/_unimol_chunks
```

**Uni-Mol2** supports both modes -- single-stage is one less command for
a quick test; two-stage is recommended once you're extracting at real
scale (see below):
```bash
# Single-stage (conformers generated inline, fine for a quick test)
python embed/compute/compute_unimol2_embeddings_chunk.py \
    --smiles-file my_smiles.txt --chunk-id 0 --num-chunks 1 \
    --chunks-dir /tmp/_unimol2_chunks

# Two-stage (recommended at real scale)
python generate_unimol2_conformers_chunk.py \
    --smiles-file my_smiles.txt --out-dir /tmp/_unimol2_conformers \
    --chunk-id 0 --num-chunks 1

python embed/compute/compute_unimol2_embeddings_chunk.py \
    --smiles-file my_smiles.txt \
    --conformer-chunks-dir /tmp/_unimol2_conformers/_chunks \
    --chunk-id 0 --num-chunks 1 --chunks-dir /tmp/_unimol2_chunks
```
Both Uni-Mol2 modes produce bit-exact identical output (verified
directly, max abs diff 0.0) -- the split only changes *where* the compute
happens, not the result.

## Stitching (after all chunks exist, any backbone)

```bash
python embed/stitch/stitch_embedding_chunks.py \
    --backbone molformer --dim 768 \
    --chunks-dir /tmp/_molformer_chunks --num-chunks 1 --total-count 3 \
    --embeddings-path /tmp/molformer_embeddings.npy
```
`--dim` per backbone: see the table above. `--num-chunks`/`--total-count`
must match whatever you used for extraction, or the stitch step will
refuse to run rather than silently misalign rows.

## At-scale extraction (SLURM, millions of molecules)

Every backbone above has a matching `slurm/embed/*/submit_*.sh` script
(grouped by dataset under `slurm/embed/ampc/` and `slurm/embed/enamine/`)
that wraps the exact same Python scripts as a `sbatch --array=...` job,
one task per chunk. Copy the closest-matching one and adjust
`--smiles-file`/`--total-count`/chunk directories for your own pool --
see any of them for the current chunk-count/timeout conventions, e.g.:
```bash
sbatch --array=0-999%40 slurm/embed/ampc/submit_ampc_grover3400_extract.sh 1000
```
For Uni-Mol2 specifically, Stage 1 (conformer generation) can run on a
plain CPU partition (no GPU needed) while Stage 2 needs GPU --
`submit_ampc_unimol2_conformers_bigred.sh` (Stage 1, CPU) and
`submit_ampc_unimol2_embed_from_conformers_h100single.sh` (Stage 2, GPU)
are separate submissions for exactly this reason.

**Measure real per-chunk throughput on a small sample before committing
to a full array** -- a fixed + variable-cost two-point timing fit (two
different chunk sizes, solve for the two constants), not a guess.
Estimates extrapolated from a different model or dataset's known
throughput have been wrong by 3x or more in practice on this exact
pipeline. `slurm/embed/ampc/submit_ampc_unimol2_conformers_timing_probe_bigred.sh`
is a worked example of this methodology (two samples, one deliberately
run twice to separate real per-molecule cost from one-time cold-cache
overhead).

## Extracting embeddings from `ibm_materials/models/`

`ibm_materials/` (a git submodule of `github.com/IBM/materials`) ships
several models; only two are wired into this pipeline's chunked-extraction
scripts so far:

| Model dir | Short name here | Status |
|---|---|---|
| `mhg_model` | `mhgged` | **wired** -- `compute_mhgged_embeddings_chunk.py` |
| `smi_ted` | `smited` | **wired** -- `compute_smited_embeddings_chunk.py` |
| `selfies_ted` | (none yet) | ready to wire -- simple SMILES/SELFIES input, real batched `encode()`, auto-fetches its own checkpoint |
| `pos_egnn` | (none yet) | ready to wire -- takes raw SMILES too, but generates a 3D conformer *internally* per molecule (slower per-molecule than the string-based models above) |
| `mol_moe`, `smi_ssed`, `3dgrid_vqgan`, `smilesdft_clip`, `str_bamba`, `tdims` | -- | more involved (task/generation-specific, not simple embedding extractors) -- not covered here |

`ibm_materials/models/fm4m.py` is IBM's own reference file showing the
canonical way to load and call every model in the table above (including
the two already wired here) -- but its usage snippets are demo-quality,
not copy-paste-safe. Everything below is verified directly against each
model's own `load.py` (not `fm4m.py`, and not assumed), because `fm4m.py`
gets both models below wrong in ways that produce silently-broken output
rather than an error.

### Walkthrough: wiring in a new backbone yourself

This is the exact recipe to follow -- worked all the way through for
SELFIES-TED and POS-EGNN below, but the same steps apply to any future
model in this directory. **Test end to end on ENHITS 2M
(`/N/project/SingleCell_Image/mengjing/enhits_large/embed/enhits_large_smiles.txt`,
2,104,318 molecules) before ever pointing at AmpC's 99.5M** -- ENHITS is
small enough that a mistake costs you a minute, not hours of GPU time.

**Step 1 -- copy an existing script, don't edit one in place.** The
production scripts (`compute_smited_embeddings_chunk.py`,
`compute_mhgged_embeddings_chunk.py`, ...) are still in real use elsewhere
(AmpC extraction, etc.) -- editing one directly to try a new model breaks
that. Copy it to a new file first:
```bash
cp embed/compute/compute_smited_embeddings_chunk.py embed/compute/compute_selfiested_embeddings_chunk.py
```
Pick whichever existing script's `compute_embeddings_for_chunk()` body is
structurally closer to the new model's `encode()`:
- **`compute_smited_embeddings_chunk.py`** -- if `encode()` does real
  batched inference and takes its own `batch_size` kwarg (SMI-TED,
  SELFIES-TED, POS-EGNN all qualify).
- **`compute_mhgged_embeddings_chunk.py`** -- if `encode()` loops
  internally one molecule at a time (`List[str] -> List[Tensor]`, no real
  batch dimension).

**Step 2 -- fix the `sys.path.insert` line to match how the new model's
package actually resolves.** This is the single most common way a
copy-paste goes wrong: **it is not the same for every model.**
`smi_ted`'s importable code (`smi_ted_light/`) lives *inside*
`ibm_materials/models/smi_ted/`, so `compute_smited_...` inserts that
specific subdirectory. `mhg_model`, `selfies_ted`, and `pos_egnn` are each
directly importable as `<dirname>.load` from the shared parent
(`ibm_materials/models` itself is a namespace package, no `__init__.py`
needed) -- confirmed directly by import-testing both:
```python
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "ibm_materials" / "models"))   # selfies_ted, pos_egnn, mhg_model
# NOT str(ROOT / "ibm_materials" / "models" / "selfies_ted") -- that pattern is smi_ted-specific
```

**Step 3 -- the model-loading + encode call**, inside
`compute_embeddings_for_chunk()`. Each model's real device-handling is
different enough that copying `smited`'s `model = model.to(DEVICE)`
pattern silently does nothing for either of these two -- verified by
reading both `load.py` files line by line, not by running `fm4m.py`'s demo:

```python
# SELFIES-TED -- ibm_materials/models/selfies_ted/load.py
from selfies_ted.load import SELFIES
model = SELFIES()
model.load()  # fetches from HF Hub (ibm/materials.selfies-ted), no local checkpoint needed
# `.to(DEVICE)` here would be a no-op that looks like it works: encode()'s
# own use_gpu arg (default False) unconditionally re-moves the model to
# CPU internally on every call, overriding whatever device you set before
# calling it. Pass use_gpu explicitly instead:
emb = model.encode(chunk_smiles, use_gpu=torch.cuda.is_available(),
                    return_tensor=True, batch_size=128)
# Row-alignment: SAFE. A SMILES that fails to convert to SELFIES gets its
# embedding row filled with NaN (see load.py's `self.invalid` handling),
# not dropped -- output row count always matches input row count.

# POS-EGNN -- ibm_materials/models/pos_egnn/load.py
from pos_egnn.load import POSEGNN
model = POSEGNN(device="cuda" if torch.cuda.is_available() else "cpu")  # device is fixed at construction, NOT via .to()
model.load()  # fetches from HF Hub (ibm-research/materials.pos-egnn), no local checkpoint needed
emb = model.encode(chunk_smiles, return_tensor=True, batch_size=32)
# Row-alignment: NOT SAFE as-is. POS-EGNN calls RDKit's EmbedMolecule() on
# each SMILES internally with no 2D-coordinate fallback (unlike this
# repo's own smiles_to_coords) -- a molecule RDKit can't embed in 3D is
# silently skipped (its load.py: `except Exception as e: print(f"Skipping
# {smiles}: {e}")`), shrinking the output row count with no marker of
# which input rows were dropped. Don't trust output row count == input
# row count -- either patch in the same NaN-fill behavior SELFIES-TED
# already does, or track which SMILES failed yourself (e.g. call
# model.encode() one small batch at a time and catch failures per-input)
# before writing the chunk file.
```

And swap every `"smited"` string literal (passed to `chunk_file_path()`/
`write_chunk_file()` in the copied file) for the new backbone's short name
(e.g. `"selfiested"`, `"posegnn"`) -- this determines the actual output
filename downstream code looks for, not just a label.

**Step 4 -- test on ENHITS 2M, one small slice first:**
```bash
python embed/compute/compute_selfiested_embeddings_chunk.py \
    --smiles-file /N/project/SingleCell_Image/mengjing/enhits_large/embed/enhits_large_smiles.txt \
    --total-count 2104318 --chunk-id 0 --num-chunks 1000 \
    --chunks-dir /tmp/_selfiested_test_chunk0
```
(`--num-chunks 1000` with `--chunk-id 0` extracts just a ~2,100-molecule
slice as a fast sanity check -- not the real full-pool run.) Load the
resulting `.npy` and confirm `matrix.shape[0] == 2104` (or whatever the
true slice size is) before trusting a bigger run -- this is exactly the
check that would have caught POS-EGNN's row-drop above immediately.

**Two more things to actually verify per new model, not assume:**
1. **Checkpoint source** -- does `load()` auto-fetch from Hugging Face Hub
   (`hf_hub_download`, like mhg-ged/smi-ted/selfies-ted/pos-egnn), or
   expect a local path under `models/` (GROVER, Uni-Mol, Uni-Mol2)? Check
   the model's own `load.py`.
2. **Embedding dim** -- don't hardcode it. Every existing script reads it
   off `matrix.shape[1]` at the end of `main()`, so whatever the model
   produces just works.

**Step 5 -- wire the new backbone into the rest of the pipeline** (one-line
additions, no logic changes):
- Add its name + dim to `embed/stitch/stitch_embedding_chunks.py`'s
  `--backbone` choices/dim mapping.
- Add its name to `run_experiment.py --backbones`'s argparse `choices`.

Nothing else needs touching -- `EmbeddingFeaturizer` looks for
`<backbone>_embeddings.npy` generically, and `LTAllSurrogate` takes `dims`
at runtime, so a new backbone is usable the moment those two lists know
its name.
