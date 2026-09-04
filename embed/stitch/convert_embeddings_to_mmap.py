#!/usr/bin/env python3
"""
One-time conversion: extract each <backbone>_embeddings.npz's 'embeddings.npy'
member out to a standalone <backbone>_embeddings.npy file.

Why this is safe/cheap: np.savez() (not savez_compressed()) stores members
with ZIP_STORED (no compression) -- each member's bytes inside the .npz ARE
already a valid standalone .npy file, byte-for-byte. So this is a streamed
disk-to-disk copy, never materializing the full (N, D) array in RAM, unlike
the old EmbeddingFeaturizer.load() path which did
np.load(npz)['embeddings'].astype(np.float32) -- a full decompress +
an unconditional copy (double-counted peak RAM) for a 13.7GB array (GROVER).

Once <backbone>_embeddings.npy exists, EmbeddingFeaturizer.load() (see
molpal/featurizer.py) prefers it and opens it with mmap_mode="r", so peak
RSS for the embedding step drops from O(N*D) per backbone (24.7GB total for
all 3 EnamineHTS backbones) to O(rows actually indexed this call).

Usage
-----
python convert_embeddings_to_mmap.py --embed-dir results/embed/EnamineHTS \
    --backbones grover molformer unimol
"""

import argparse
import shutil
import zipfile
from pathlib import Path


def convert_one(npz_path: Path, npy_path: Path, member: str = "embeddings.npy"):
    with zipfile.ZipFile(npz_path) as z:
        info = z.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{npz_path}:{member} is compressed (compress_type="
                f"{info.compress_type}); this script only handles the "
                f"ZIP_STORED case produced by np.savez(). Falling back to "
                f"a full np.load() decompress would defeat the point."
            )
        tmp_path = npy_path.with_suffix(".npy.tmp")
        with z.open(member) as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
        tmp_path.rename(npy_path)
    print(f"[convert] {npz_path.name} -> {npy_path.name}  "
          f"({npy_path.stat().st_size / 1e9:.2f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", required=True, type=Path)
    ap.add_argument("--backbones", nargs="+", required=True)
    args = ap.parse_args()

    for bb in args.backbones:
        npz_path = args.embed_dir / f"{bb}_embeddings.npz"
        npy_path = args.embed_dir / f"{bb}_embeddings.npy"
        if not npz_path.exists():
            print(f"[skip] {npz_path} not found")
            continue
        if npy_path.exists():
            print(f"[skip] {npy_path} already exists")
            continue
        convert_one(npz_path, npy_path)


if __name__ == "__main__":
    main()
