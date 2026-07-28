#!/usr/bin/env bash
set -euo pipefail

# score-fragments.sh — score all best-fit RNA fragments in the current
# directory against one receptor, with attract-jax minfor.py.
#
# Detects every frag-X-*-bestfit.pdb file in the current directory (X = the
# fragment number). The receptor Y.pdb is parsed/reduced once and used to
# build a single NB potential grid, which is then reused to score each
# fragment (identity transform: each bestfit PDB is scored exactly as
# positioned in its file). One frag-X-Y.score file is written per fragment.
#
# Usage:
#   bash score-fragments.sh Y DATADIR
#
# Arguments:
#   Y         basename (without .pdb) of the receptor PDB; DATADIR/Y.pdb must
#             exist (e.g. 1b7f_dom1-aa.pdb -> Y=1b7f_dom1-aa)
#   DATADIR   directory containing Y.pdb
#
# Output:
#   frag-X-Y.score written to the current directory for each detected
#   frag-X-*-bestfit.pdb, containing the single-line energy value.

usage() {
  echo "Usage: bash score-fragments.sh Y DATADIR" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

Y=$1
DATADIR=$2

receptor_pdb="${DATADIR}/${Y}.pdb"
if [[ ! -f "${receptor_pdb}" ]]; then
  echo "Receptor PDB not found: ${receptor_pdb}" >&2
  exit 1
fi

shopt -s nullglob
frag_files=(frag-*-bestfit.pdb)
if [[ ${#frag_files[@]} -eq 0 ]]; then
  echo "No frag-*-bestfit.pdb files found in $(pwd)" >&2
  exit 1
fi

XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
ATTRACT_PAR_NPZ="${SCRIPT_DIR}/../attract-jax/attract-original/attract-par.npz"
NB_KERNEL="compiled"

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT INT TERM

tmp_rec_ppdb="${tmpdir}/receptor-ppdb.npy"
tmp_rec_prefix="${tmpdir}/receptor"
tmp_rec_coor="${tmp_rec_prefix}-coor.npy"
tmp_rec_atomtypes="${tmp_rec_prefix}-atomtypes.npy"

tmp_lig_ppdb="${tmpdir}/ligand-ppdb.npy"
tmp_lig_prefix="${tmpdir}/ligand"
tmp_lig_coor="${tmp_lig_prefix}-coor.npy"
tmp_lig_atomtypes="${tmp_lig_prefix}-atomtypes.npy"

tmp_grid_atomtypes="${tmpdir}/grid-ligand-atomtypes.txt"
tmp_grid="${tmpdir}/receptor.grid.npz"
tmp_energy="${tmpdir}/energy.npy"

# --- Step 1: parse and reduce receptor PDB -> ATTRACT beads ---
echo "Parsing receptor PDB: ${receptor_pdb}" >&2
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/parse_pdb.py" \
  "${receptor_pdb}" "${tmp_rec_ppdb}"
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/reduce-npy.py" \
  "${tmp_rec_ppdb}" "${tmp_rec_prefix}"

# --- Step 2: build one NB potential grid for the receptor ---
# The ligand-atomtypes hint covers every valid ATTRACT bead type (1-98) so
# the grid supports any fragment scored against it later, regardless of
# which nucleotide beads that fragment happens to contain.
seq 1 98 > "${tmp_grid_atomtypes}"

echo "Building NB potential grid for ${receptor_pdb}..." >&2
"${PYTHON}" -u "${SCRIPT_DIR}/../attract-jax/util/minfor.py" \
  --generate-grid \
  --grid "${tmp_grid}" \
  --oracle jax \
  --attract-par-npz "${ATTRACT_PAR_NPZ}" \
  --nb-kernel "${NB_KERNEL}" \
  --receptor-coordinates "${tmp_rec_coor}" \
  --receptor-atomtypes "${tmp_rec_atomtypes}" \
  --ligand-atomtypes "${tmp_grid_atomtypes}"

# --- Step 3: score each fragment against the receptor grid ---
for frag_pdb in "${frag_files[@]}"; do
  if [[ ! "${frag_pdb}" =~ ^frag-([0-9]+)-.+-bestfit\.pdb$ ]]; then
    echo "Skipping ${frag_pdb}: does not match frag-X-*-bestfit.pdb" >&2
    continue
  fi
  X="${BASH_REMATCH[1]}"
  score_file="frag-${X}-${Y}.score"

  echo "Parsing fragment PDB: ${frag_pdb} (X=${X})" >&2
  PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" "${SCRIPT_DIR}/parse_pdb.py" \
    "${frag_pdb}" "${tmp_lig_ppdb}"
  PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" "${SCRIPT_DIR}/reduce-npy.py" \
    "${tmp_lig_ppdb}" "${tmp_lig_prefix}"

  echo "Scoring ${frag_pdb} against ${receptor_pdb}..." >&2
  cmd=(
    "${PYTHON}" -u "${SCRIPT_DIR}/../attract-jax/util/minfor.py"
    --identity
    --score
    --energy-only
    --oracle jax
    --attract-par-npz "${ATTRACT_PAR_NPZ}"
    --nb-kernel "${NB_KERNEL}"
    --output-npy "${tmp_energy}"
    --score-mode bulk
    --no-conformer-grouping
    --receptor-coordinates "${tmp_rec_coor}"
    --receptor-atomtypes "${tmp_rec_atomtypes}"
    --grid "${tmp_grid}"
    --ligand-ensemble "${tmp_lig_coor}"
    --ligand-atomtypes "${tmp_lig_atomtypes}"
  )
  env XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}" \
    "${cmd[@]}" >&2

  PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -c 'import sys, numpy as np; e = float(np.load(sys.argv[1]).reshape(-1)[0]); print(repr(e))' \
    "${tmp_energy}" > "${score_file}"

  echo "Wrote ${score_file}" >&2
done
