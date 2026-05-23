#!/usr/bin/env bash
set -euo pipefail

# score.sh — score alaric poses with attract-jax minfor
#
# Usage:
#   bash score.sh POSE_DIR FIRST_INDEX LAST_INDEX SEQUENCE \
#                 RECEPTOR_PDB LIGAND_ENSEMBLE LIGAND_ATOMTYPES \
#                 NB_KERNEL [OUTPUT_NPY]
#
# Arguments:
#   POSE_DIR         directory with poses-<index>.arc[.zst] files
#   FIRST_INDEX      first inclusive shard index (e.g. 1)
#   LAST_INDEX       last  inclusive shard index (e.g. 1)
#   SEQUENCE         dinucleotide sequence, e.g. GU
#   RECEPTOR_PDB     all-atom receptor PDB (e.g. 1b7f_dom2-aa.pdb); will be
#                    parsed with parse_pdb.py and reduced to ATTRACT beads
#   LIGAND_ENSEMBLE  reduced ligand coordinate .npy, shape (C, N, 3)
#   LIGAND_ATOMTYPES per-atom ATTRACT type .npy
#   NB_KERNEL        'compiled' or 'jax'
#   OUTPUT_NPY       output energies .npy (default: energies.npy)

pose_dir=$1
first_index=$2
last_index=$3
sequence=$4
receptor_pdb=$5
ligand_ensemble=$6
ligand_atomtypes=$7
nb_kernel=$8
output_file=${9:-energies.npy}

SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# Auto-generate attract-par.npz if not present
if [[ ! -f "${SCRIPT_DIR}/attract-par.npz" ]]; then
  echo "Generating attract-par.npz from attract.par..." >&2
  (cd "${SCRIPT_DIR}/attract-jax/attract-original" && \
   "${PYTHON}" convert-attract-par.py)
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT INT TERM

if (( first_index > last_index )); then
  echo "first_index must be <= last_index" >&2
  exit 1
fi

tmp_prefix="${tmpdir}/poses-${first_index}"
if (( first_index != last_index )); then
  tmp_prefix+="-${last_index}"
fi
tmp_rotvec="${tmp_prefix}.rotvec.npy"
tmp_conformers="${tmp_prefix}.conformers.npy"
tmp_score="${tmpdir}/score.out"

tmp_ppdb="${tmpdir}/receptor-ppdb.npy"
tmp_rec_prefix="${tmpdir}/receptor"
tmp_rec_coor="${tmp_rec_prefix}-coor.npy"
tmp_rec_atomtypes="${tmp_rec_prefix}-atomtypes.npy"

# --- Step 1: parse and reduce receptor PDB → ATTRACT beads ---
echo "Parsing receptor PDB: ${receptor_pdb}" >&2
PYTHONPATH="${SCRIPT_DIR}/code${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/code/parse_pdb.py" \
  "${receptor_pdb}" "${tmp_ppdb}"

echo "Reducing receptor to ATTRACT beads..." >&2
PYTHONPATH="${SCRIPT_DIR}/code${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/util/reduce-npy.py" \
  "${tmp_ppdb}" "${tmp_rec_prefix}"

# --- Step 2: convert alaric arc poses → rotvec DOFs ---
echo "Converting poses to rotvec DOFs..." >&2
t_convert_start=$(date +%s%N)
PYTHONPATH="${SCRIPT_DIR}/code${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/code/convert_poses.py" \
  --pose-dir "${pose_dir}" \
  --first-index "${first_index}" \
  --last-index "${last_index}" \
  --sequence "${sequence}" \
  --output-prefix "${tmp_prefix}"
t_convert_end=$(date +%s%N)
t_convert_ms=$(( (t_convert_end - t_convert_start) / 1000000 ))
echo "convert_poses.py finished in ${t_convert_ms} ms" >&2

# --- Step 3: score with minfor.py ---
echo "Scoring with minfor.py (rotvec)..." >&2
t_score_start=$(date +%s%N)

cmd=(
  "${PYTHON}" -u "${SCRIPT_DIR}/attract-jax/util/minfor.py"
  --input-rotvec "${tmp_rotvec}"
  --input-conformers "${tmp_conformers}"
  --input-world-centered
  --score
  --energy-only
  --oracle jax
  --attract-par-npz "${SCRIPT_DIR}/attract-par.npz"
  --nb-kernel "${nb_kernel}"
  --output-npy "${output_file}"
  --score-mode bulk
  --receptor-coordinates "${tmp_rec_coor}"
  --receptor-atomtypes "${tmp_rec_atomtypes}"
  --ligand-ensemble "${ligand_ensemble}"
  --ligand-atomtypes "${ligand_atomtypes}"
)
if [[ -n "${SCORE_BATCH_SIZE}" ]]; then
  cmd+=(--score-batch-size "${SCORE_BATCH_SIZE}")
fi

env XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}" \
  "${cmd[@]}" > "${tmp_score}"

t_score_end=$(date +%s%N)
t_score_ms=$(( (t_score_end - t_score_start) / 1000000 ))
echo "minfor.py (score) finished in ${t_score_ms} ms" >&2
