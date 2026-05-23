#!/usr/bin/env bash
set -euo pipefail

# score.sh — score alaric poses with attract-jax minfor
#
# Usage:
#   bash score.sh [-x PDB] ... POSE_DIR POSE_START POSE_END SEQUENCE \
#                 RECEPTOR_PDB NB_KERNEL [OUTPUT_NPY]
#
# Options:
#   -x PDB           exclude this PDB code from the fragment library (repeatable)
#
# Arguments:
#   POSE_DIR         directory with poses-*.arc[.zst] files
#   POSE_START       first pose (1-based, global across all shards)
#   POSE_END         last pose  (1-based, end-inclusive)
#   SEQUENCE         dinucleotide sequence, e.g. GU
#   RECEPTOR_PDB     all-atom receptor PDB (e.g. 1b7f_dom2-aa.pdb); will be
#                    parsed with parse_pdb.py and reduced to ATTRACT beads
#   NB_KERNEL        'compiled' or 'jax'
#   OUTPUT_NPY       output energies .npy (default: energies.npy)

exclude_pdbs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -x|--exclude)
      exclude_pdbs+=("$2")
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

pose_dir=$1
pose_start=$2
pose_end=$3
sequence=$4
receptor_pdb=$5
nb_kernel=$6
output_file=${7:-energies.npy}

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

if (( pose_start > pose_end )); then
  echo "POSE_START must be <= POSE_END" >&2
  exit 1
fi

tmp_rotvec="${tmpdir}/poses.rotvec.npy"
tmp_conformers="${tmpdir}/poses.conformers.npy"
tmp_score="${tmpdir}/score.out"

tmp_ppdb="${tmpdir}/receptor-ppdb.npy"
tmp_rec_prefix="${tmpdir}/receptor"
tmp_rec_coor="${tmp_rec_prefix}-coor.npy"
tmp_rec_atomtypes="${tmp_rec_prefix}-atomtypes.npy"

tmp_ligand_ensemble="${tmpdir}/ligand-ensemble.npy"
tmp_ligand_atomtypes="${tmpdir}/ligand-atomtypes.npy"

# --- Step 1: parse and reduce receptor PDB → ATTRACT beads ---
echo "Parsing receptor PDB: ${receptor_pdb}" >&2
PYTHONPATH="${SCRIPT_DIR}/code${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/code/parse_pdb.py" \
  "${receptor_pdb}" "${tmp_ppdb}"

echo "Reducing receptor to ATTRACT beads..." >&2
PYTHONPATH="${SCRIPT_DIR}/code${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/util/reduce-npy.py" \
  "${tmp_ppdb}" "${tmp_rec_prefix}"

# --- Step 2: build ligand ensemble and atomtypes from fragment library ---
echo "Building ligand ensemble (sequence=${sequence}, exclude=[${exclude_pdbs[*]:-none}])..." >&2
exclude_flags=()
for pdb in "${exclude_pdbs[@]}"; do
  exclude_flags+=(-x "${pdb}")
done
PYTHONPATH="${SCRIPT_DIR}/code${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/util/get-reduced-library.py" \
  "${sequence}" "${tmp_ligand_ensemble}" \
  "${exclude_flags[@]}"

PYTHONPATH="${SCRIPT_DIR}/code${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/util/ligand-atomtypes.py" \
  --sequence "${sequence}" "${tmp_ligand_atomtypes}"

# --- Step 3: convert alaric arc poses → rotvec DOFs ---
echo "Converting poses to rotvec DOFs..." >&2
t_convert_start=$(date +%s%N)
PYTHONPATH="${SCRIPT_DIR}/code${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/code/convert_poses.py" \
  --pose-dir "${pose_dir}" \
  --pose-range "${pose_start}" "${pose_end}" \
  --sequence "${sequence}" \
  --output-prefix "${tmpdir}/poses"
t_convert_end=$(date +%s%N)
t_convert_ms=$(( (t_convert_end - t_convert_start) / 1000000 ))
echo "convert_poses.py finished in ${t_convert_ms} ms" >&2

# --- Step 4: score with minfor.py ---
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
  --ligand-ensemble "${tmp_ligand_ensemble}"
  --ligand-atomtypes "${tmp_ligand_atomtypes}"
)
cmd+=(--no-conformer-grouping)
if [[ -n "${NO_NB_GRID_EVAL:-}" ]]; then
  cmd+=(--no-nb-grid-eval)
fi
if [[ -n "${NO_POT_GRID_EVAL:-}" ]]; then
  cmd+=(--no-pot-grid-eval)
fi
if [[ -n "${SCORE_BATCH_SIZE}" ]]; then
  cmd+=(--score-batch-size "${SCORE_BATCH_SIZE}")
fi

env XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}" \
  "${cmd[@]}" > "${tmp_score}"

t_score_end=$(date +%s%N)
t_score_ms=$(( (t_score_end - t_score_start) / 1000000 ))
t_total_ms=$(( (t_score_end - t_convert_start) / 1000000 ))
echo "minfor.py (score) finished in ${t_score_ms} ms" >&2
echo "Total (convert + score): ${t_total_ms} ms" >&2
