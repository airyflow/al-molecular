#!/bin/bash

#SBATCH -J lsd_download
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -p general
#SBATCH -o logs/lsd_download_%j.txt
#SBATCH -e logs/lsd_download_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH -A r00939

# Downloads one screen's DockingResults from the LSD database
# (lsd.docking.org / files.docking.org/lsd), an open HTTP directory -- no
# login or token. Each screen is split into a few hundred ~350 MB CSV part
# files with the header "smiles,zincid,score"; DOCK3.7/3.8 scores, the same
# scoring function as this project's existing AmpC and D4 pools, so recall
# numbers are comparable across them (unlike the Glide HTVS scores of the
# Enamine REAL / HASTEN pool).
#
# Sizes (checked 2026-10-13):
#   AmpC/Liu_2025   294 parts, ~106 GB   1,568,323,216 molecules
#   5HT2A/Lyu_2024  312 parts, ~108 GB   1,630,264,067 molecules
#   AmpC/Lyu_2019                        the 99.5M screen already in use here
#
# Parallelism is deliberately modest (8 concurrent transfers): this is a
# public academic server, not a CDN. curl -C - resumes partial files, so
# re-running the script after an interruption only fetches what is missing.
# Files are NOT verified against checksums -- the server publishes none --
# so the script re-checks each file's size against the server's
# Content-Length and reports any mismatch at the end.
#
# Usage:
#   sbatch slurm/embed/lsd/download_lsd_screen.sh AmpC/Liu_2025
#   sbatch slurm/embed/lsd/download_lsd_screen.sh 5HT2A/Lyu_2024
#   bash   slurm/embed/lsd/download_lsd_screen.sh AmpC/Liu_2025   # also fine

set -euo pipefail

SCREEN="${1:?Usage: $0 <Target>/<Screen>, e.g. AmpC/Liu_2025}"
BASE="https://files.docking.org/lsd/$SCREEN/DockingResults"
DEST="/N/project/SingleCell_Image/mengjing/lsd/${SCREEN//\//_}"
JOBS=8

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then cd "$SLURM_SUBMIT_DIR"; fi
mkdir -p logs "$DEST"
cd "$DEST"

echo "[lsd] screen=$SCREEN"
echo "[lsd] dest=$DEST"
curl -sSL "https://files.docking.org/lsd/$SCREEN/reference.txt" | head -5 || true

# Part-file names come from the server listing, not assumed (the split
# suffixes are not contiguous across screens).
curl -sSL "$BASE/" \
  | grep -o 'href="[^"?][^"]*\.csv\(\.gz\)\?"' \
  | sed 's/href="//; s/"$//' \
  | sort -u > _parts.txt
N=$(wc -l < _parts.txt)
echo "[lsd] $N part files listed"
[ "$N" -gt 0 ] || { echo "[lsd] ERROR: no part files found at $BASE/"; exit 1; }

echo "[lsd] downloading with $JOBS concurrent transfers (resume enabled)"
xargs -a _parts.txt -I{} -P "$JOBS" \
    curl -sS -L -C - --retry 5 --retry-delay 20 -o {} "$BASE/{}"

echo "[lsd] verifying sizes against server Content-Length"
bad=0
while read -r f; do
    want=$(curl -sSLI "$BASE/$f" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -1)
    got=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$want" != "$got" ]; then echo "[lsd] SIZE MISMATCH $f: have $got want $want"; bad=$((bad+1)); fi
done < _parts.txt

total=$(du -sh . | cut -f1)
if [ "$bad" -eq 0 ]; then
    echo "[lsd] done: $N/$N files complete, $total"
else
    echo "[lsd] done with $bad incomplete file(s) -- re-run this script to resume them"
    exit 1
fi
