# Scaling al-molecular to 1.3B molecules: oracle memory, dedup, and parallel PCA

Oct 7, 2026

## Context and goals

Three parts of the LT-All pipeline keep one record per pool molecule in RAM or in one process; at 1.3B molecules they need about 0.6–1 TB of memory or ~9 days of single-process time. This doc replaces them with row-indexed arrays, a hash-partitioned dedup, and a row-range-parallel PCA transform. Everything else (sharded prediction, marker-file coordination, sequential block reads, resume) already scales and is unchanged.

Scale reference: AmpC deduplicated = 98,489,350 rows; D4 deduplicated = 116,225,927 rows; target = ~1.3B rows (13× AmpC, 11× D4).

| Fix | What breaks today | Where | At 1.3B (extrapolated) |
| --- | --- | --- | --- |
| 1. Oracle and pool memory | SMILES-keyed dicts and full SMILES arrays held by the orchestrator and by every LT-All worker | `run_experiment.py`, `molpal/models/mvemodels.py`, `predict_pool_shard_worker.py` | Oracle dict alone ~24 GB at 98.5M → ~320 GB; several similar structures on top |
| 2. Dedup keep-mask | One Python dict over every SMILES string | `embed/stitch/build_dedup_mask.py` | D4 job asked for 64 GB at 116M → ~700 GB |
| 6. PCA transform | Sampling scan and transform each read the whole file in one process | `embed/stitch/pca_reduce_embeddings.py` | GROVER3400: ~60,500 s of full-file passes at 98.5M → ~222 h |

Non-goals: MPN at 1.3B (training cost, not memory, is the blocker), embedding extraction cost, and raw-embedding storage (assumed solved by storing PCA-reduced embeddings, ~3.4 TB at 653-d fused). Fixes 3–5 from the scoping discussion (storage, labeled-embedding cache, per-round fetch) are out of scope here; they mostly fall away once PCA-reduced embeddings are the default input.

## Fix 1: row index as the only molecule key

At runtime a molecule is its row number; scores live in a float32 array aligned to the library rows; SMILES are read from disk only to write the final CSV. This cuts the orchestrator's pool-sized memory at 1.3B from several hundred GB to about 40 GB at peak, and removes all SMILES loading from LT-All workers.

### What holds pool-sized memory today (LT-All parallel path)

| Structure | Built in | Notes |
| --- | --- | --- |
| SMILES → score dict | `load_oracle()` | ~24 GB at 98.5M (measured, report §11); ~320 GB at 1.3B |
| Full SMILES list | `load_library_smiles()` via `load_embeddings()` | Python strings, not measured |
| Fixed-width SMILES array | `MVEExplorer.__init__`: `np.array(pool_smiles)` | 4 bytes × longest SMILES, per row |
| SMILES → row dict | `EmbeddingMVEModel.smi2idx` | Same order of size as the oracle dict |
| Candidate mask | `main()`: generator with one set lookup per row | Result is 1.3 GB; the build is the slow part |
| True top-k | `true_top_k_set()`: Python sort of all (SMILES, score) pairs | Full-size transient list |
| Full SMILES list, per worker | `predict_pool_shard_worker.py` calls `load_embeddings()`, then slices its shard | Paid 8× (once per worker) |

### Design

1. **One-time prep script** `embed/stitch/build_pool_arrays.py`. It streams the deduplicated scores file and writes `scores.npy` (float32, shape N, NaN where unscored). It also streams the library file alongside it and asserts row i's SMILES matches in both. The scores file is already row-aligned with the SMILES file for AmpC (`dedup_smiles_and_scores.py` docstring) and D4 (report §16); the 1.3B source must be produced the same way.
2. **Orchestrator state, all arrays:**
   - `scores` = `np.load(..., mmap_mode='r')`, 5.2 GB.
   - `usable_mask` = `~np.isnan(scores)`, 1.3 GB, vectorized.
   - `topk_mask` from `np.argpartition` on scores (NaN → +inf), O(N); about 10 GB transient while building, then a k-row index array.
   - `labeled_mask` (bool, 1.3 GB) replaces the SMILES-keyed `labeled_scores` dict. `_best()` = `scores[labeled].min()`; `_recall()` = `topk_mask[labeled].sum() / k`.
3. **Per-round selection:** set acquisition scores to -inf where `usable_mask & ~labeled_mask` is False, then `np.argpartition` for the top B. This replaces `np.where` + a full `argsort` over the remaining pool, and replaces the Python loop `for i in labeled_idx: mask[i] = False`.
4. **Embedding model by row range:** add a range-based predict to `EmbeddingMVEModel` (`_get_X_idx(start, end)`) and skip building `smi2idx`. On a deduplicated pool row and molecule are 1:1, so the duplicate-SMILES patching in `_get_X` is no longer needed on this path.
5. **Workers:** memory-map the embeddings, take `[start, end)` from `_chunk_bounds`, predict in contiguous ranges. No SMILES list is loaded.
6. **Checkpoints and output:** `iter_N/labeled_idx.npy` (int64) replaces `iter_N/scores.pkl`. `all_explored_final.csv` keeps its current format: one streaming pass over the library at the end writes SMILES + score for labeled rows. Resume falls back to `scores.pkl` when `labeled_idx.npy` is missing (runs started before this change, ≤116M rows only).

### Memory at 1.3B, orchestrator

| Structure | Today | Proposed |
| --- | --- | --- |
| Scores | ~320 GB dict | 5.2 GB memmap |
| Pool SMILES (list, array, row dict) | not measured; each comparable to the dict | not loaded |
| Candidate and labeled masks | 1.3 GB + slow build | 2.6 GB, vectorized |
| True top-k | full Python sort | ~10 GB transient |
| Per-round mu and var | 10.4 GB | 10.4 GB (unchanged) |

The labeled-embedding cache is not in this table; it is sized by the acquisition fraction and embedding width (fix 4, out of scope).

Scope: only the parallel LT-All path (`ParallelMVEExplorer` and `predict_pool_shard_worker.py`) switches. The single-process `MVEExplorer` and MolPAL paths only run on pools of 2M or less and stay as they are, so published EnamineHTS results are untouched.

## Fix 2: hash-partitioned dedup keep-mask

The new job hashes every SMILES to 128 bits, sorts the (hash, row) pairs into 256 partition files, then keeps the largest row of each hash within each partition. Each task needs well under 1 GB of memory, and the output keep-mask is the same file format today's `build_dedup_mask.py` writes.

Today's rule stays exactly the same: keep the last occurrence of each exact SMILES string. Today that comes from a dict overwrite (`last_occurrence[s] = i`); here it is the maximum row in each equal-hash group.

### Steps

1. **Scan (one task).** Stream the library file. For each row, compute `blake2b(line_bytes, digest_size=16)` and write a 24-byte record (hash high 64 bits, hash low 64 bits, row) to partition `hash_high >> 56`. Records are buffered and flushed per partition. On-disk total ≈ 24 B × 1.3B ≈ 31 GB, about 122 MB per partition.
2. **Reduce (array job, 256 tasks).** Each task loads one partition (~5.1M records), sorts by (hash high, hash low, row) with `np.lexsort`, and keeps the last row of each equal-hash run. Output: `keep_rows_<p>.npy`.
3. **Merge (one task).** Allocate `keep = np.zeros(N, bool)` (1.3 GB), set `keep[rows] = True` for every partition's rows, and save. Assert that the count of `True` equals the total number of distinct hashes.

### Hash width

128 bits, because a 64-bit hash is not safe at this size. For n distinct SMILES, the chance that any two collide is about n² / 2^(bits+1). At n = 1.3B that is about 5% for 64 bits and about 2.5 × 10⁻²¹ for 128 bits. A collision would silently drop a real molecule, so 128 bits is used and no separate string check is done.

### Time

Not measured. The scan is a Python line loop plus one hash per line, so expect very roughly 15–60 minutes for 1.3B rows; measure on D4 first. If it is too slow, split the scan by byte range at line boundaries. Each task then writes local row numbers plus its own line count, and the reduce step adds each task's row offset.

### Unchanged

`dedup_smiles_and_scores.py` and `dedup_embeddings.py` already stream their inputs row by row against the mask, so they need no change at 1.3B. They only need the 1.3 GB mask in memory.

## Fix 6: parallel PCA by row range

`pca_reduce_embeddings.py` becomes three SLURM stages chained with `--dependency=afterok`: a parallel sample-gather, a single fit, and a parallel transform. For GROVER3400 at 1.3B rows the two full-file passes go from about 222 h (~9 days) in one process to about 7 h across 32 tasks. That assumes aggregate read bandwidth scales with task count, which is not measured.

### Why both full-file passes must split

The fit itself does not grow with N, because it uses a fixed sample. For GROVER3400 it was 10,239 s on 441,176 rows after the LAPACK cap. But both the sampling scan and the transform read the whole source file. On AmpC (98.5M rows), GROVER3400's sampling scan took about 12,850 s (extrapolated from its last progress line) and its transform 47,640 s, about 60,500 s together; scaled 13.2× that is about 798,000 s (~222 h). The log's final "total" time covers only the transform, which an earlier version of this doc misread as the whole run. The sample is ~1M rows spread over 1.3B, so nearly every 500,000-row block holds a sampled row; skipping blocks without hits saves nothing.

### Stages

1. **Gather (array job, T tasks).** Each task regenerates the same `sample_idx` from `np.random.default_rng(seed)` and keeps the indices that fall in its own row range from `_chunk_bounds(N, i, T)`. It streams that range and writes `sample_part_<i>.npy`. The union of all parts is row-for-row the sample the current single-process code draws, because the indices come from the same seed and the same draw.
2. **Fit (one task).** Concatenate the parts in task order, fit with the same solver and the same LAPACK element cap, and write `.pca_params.npz` as today. Then call `preallocate(out_path, N, k)` once, so no transform task races to create the file.
3. **Transform (array job, T tasks).** Each task loads the params, takes its `[start, end)` range, and streams it in 500,000-row blocks with `write_slice`. It fails fast if the output file is missing or has the wrong shape.

### Resume

`_find_resume_point` gains `start` and `end` arguments and binary-searches only inside its own range. Writes within one range stay strictly in row order, so its "last non-zero row" logic still holds. A resubmitted array task redoes only its unfinished part; the existing `--array=<i>` resubmit pattern applies unchanged.

### Concurrent writes

`preallocate` and `write_slice` in `shared_embedding_store.py` were written for several tasks filling disjoint row ranges of one preallocated `.npy`, so no new write mechanism is needed. One risk to check: the `.npy` header and row widths do not line up with filesystem pages, so two neighbouring ranges can share one page at their boundary. The validation plan compares the parallel output byte for byte against the serial output to rule this out.

### Sizing

T = 32 is a starting point, not a measured optimum. D4's GROVER3400 embedding-cache read streamed about 1.6 GB/s in one process. If 32 concurrent readers do not get close to 32× that, fewer, longer tasks are just as fast. Each task's memory is one block plus the params: about 7 GB for GROVER3400 (500,000 × 3,400 × 4 bytes), much less for the other backbones.

## Validation plan

Each fix must reproduce today's outputs exactly on AmpC and D4 before it touches anything bigger. After that, one end-to-end run on a ~300M-row slice is the gate before 1.3B.

| Fix | Check | Pass criterion |
| --- | --- | --- |
| 1 | `build_pool_arrays.py` on AmpC_dedup and D4_dedup | Every row's SMILES matches between scores and library files; `usable_mask.sum()` equals today's `[pool]` log count (D4: 116,225,927) |
| 1 | True top-k by argpartition vs. `true_top_k_set()` | Same row set for k = 50,000 on both pools |
| 1 | Replay D4 LT-All frac=0.004: feed the saved per-shard mu/var files and each previous round's labeled set into the new selection code | Same init set; same selected rows every round; same recall as report §16 (32.0, 55.7, 66.6, 72.7, 76.9%) |
| 1 | Worker and orchestrator peak RSS (`sstat MaxRSS`) on that rerun | Orchestrator well under today's 280 GB request; workers hold no SMILES list |
| 2 | New dedup on AmpC raw (99,459,561 rows) and D4 raw (116,241,184 rows) | `np.array_equal` with `ampc_99.5M/dedup/keep_mask.npy` and `d4_138M/d4_dedup_keep_mask.npy` |
| 6 | Parallel PCA on AmpC GROVER3400 and MolFormer, T = 8 | Same `sample_idx`; `.pca_params.npz` equal to the existing files (32 and 271 components); output `.npy` byte-identical to `ampc_99.5M/dedup_pca/` |

The replay is the exact test for fix 1: it isolates selection from GPU training, which may not be bit-reproducible. Any mismatch in selected rows is a bug, with one exception: `np.argpartition` and today's full `argsort` can order exactly tied acquisition scores differently, so a mismatch is acceptable only if it sits on such a tie. A full new-path rerun is then a smoke test, expected to land within normal run-to-run variation of 76.9%, not to match exactly.

Then the 300M-row slice: the first 300M rows of the 1.3B library through dedup, `build_pool_arrays.py`, PCA, and a 5-round LT-All run at frac=0.001. Record wall time and peak memory per stage and use them to size the 1.3B jobs, instead of the extrapolations above.

## Rollout, risks, and open questions

Build the three fixes in data-flow order: dedup, then PCA, then the row-indexed run. Each one is checked on AmpC and D4 before the next starts.

1. **Fix 2 (dedup).** Smallest change, no runtime code touched, and the first step any new pool goes through.
2. **Fix 6 (parallel PCA).** Changes only the reduction script; existing reduced files are the reference outputs.
3. **Fix 1 (row-indexed run).** Largest change, touching the orchestrator, the embedding model, the workers, and the checkpoint format; it goes last so it is tested on inputs already produced by fixes 2 and 6.
4. **300M-row slice end to end**, then 1.3B, with job sizes taken from the slice's measured times.

### Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| 1.3B scores file not row-aligned with its SMILES file | `build_pool_arrays.py` refuses to run | Ask for an aligned file; otherwise add a sort-merge join (not designed here) |
| Shared read bandwidth does not scale to 32 concurrent PCA tasks | Fix 6 saves less time than estimated | Measure on the slice; use fewer, longer tasks |
| Neighbouring transform ranges share a page at their boundary | Corrupt rows near range edges | Byte-compare against serial output (validation plan) |
| Per-user limit of 1,000 submitted array tasks per partition | Large stages can't be submitted all at once | 256 reduce tasks plus 6 backbones × 64 PCA tasks = 640; submit in waves |
| Checkpoint format changes in fix 1 | Older runs can't resume on the new path | `scores.pkl` fallback for runs of 116M rows or fewer |

Temporary disk needed: about 31 GB of dedup partition files and about 6 GB of GROVER3400 sample parts, both deleted after their stage.

### Open questions

- [ ] In what format will the 1.3B library arrive, and is its scores file row-aligned with the SMILES file?
- [ ] Top-k at 1.3B: keep k = 50,000, or scale to the same fraction of the pool (~650,000)? This decides whether recall compares with AmpC and D4.
- [ ] Acquisition fraction for 1.3B: at 0.4% the labeled set reaches ~31M rows; this sizes the labeled-embedding cache (fix 4, not covered here).
