#!/usr/bin/env bash
set -euo pipefail

# score-single.sh — score a single receptor/ligand complex with attract-jax
#
# Unlike score.sh (which scores many alaric poses from a fragment library),
# this scores one receptor PDB against one ligand PDB, taken exactly as they
# are positioned in their files (identity transform). The energy is printed
# to stdout.
#
# Usage:
#   bash score-single.sh RECEPTOR_PDB LIGAND_PDB
#
# Arguments:
#   RECEPTOR_PDB     all-atom receptor PDB (e.g. 1b7f_dom2-aa.pdb); parsed with
#                    parse_pdb.py and reduced to ATTRACT beads
#   LIGAND_PDB       all-atom ligand PDB; parsed and reduced the same way

usage() {
  echo "Usage: bash score-single.sh RECEPTOR_PDB LIGAND_PDB" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

receptor_pdb=$1
ligand_pdb=$2

XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
ATTRACT_PAR_NPZ="${SCRIPT_DIR}/attract-jax/attract-original/attract-par.npz"
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

tmp_energy="${tmpdir}/energy.npy"

# --- Step 1: parse and reduce receptor PDB → ATTRACT beads ---
echo "Parsing receptor PDB: ${receptor_pdb}" >&2
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/parse_pdb.py" \
  "${receptor_pdb}" "${tmp_rec_ppdb}"
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/util/reduce-npy.py" \
  "${tmp_rec_ppdb}" "${tmp_rec_prefix}"

# --- Step 2: parse and reduce ligand PDB → ATTRACT beads ---
echo "Parsing ligand PDB: ${ligand_pdb}" >&2
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/parse_pdb.py" \
  "${ligand_pdb}" "${tmp_lig_ppdb}"
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/util/reduce-npy.py" \
  "${tmp_lig_ppdb}" "${tmp_lig_prefix}"

# --- Step 3: score the single complex with minfor.py (identity transform) ---
# The reduced ligand bead coordinates (N,3) are used directly as a
# single-conformer ligand ensemble; --identity scores it where it sits.
echo "Scoring with minfor.py (identity)..." >&2
cmd=(
  "${PYTHON}" -u "${SCRIPT_DIR}/attract-jax/util/minfor.py"
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
  --ligand-ensemble "${tmp_lig_coor}"
  --ligand-atomtypes "${tmp_lig_atomtypes}"
)

env XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}" \
  "${cmd[@]}" >&2

# --- Step 4: print the energy to stdout ---
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -c 'import sys, numpy as np; e = np.load(sys.argv[1]).reshape(-1); print(" ".join(repr(float(x)) for x in e))' \
  "${tmp_energy}"
