#!/usr/bin/env bash
# Run from any directory; the checkout supplies Alaric source, not a pip install.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: refe-best-fit-SEAMLESS.sh [WRAPPER_OPTIONS] RNA.pdb [FITTER_OPTIONS]

Wrapper options (before RNA.pdb):
  --output DIR           Download TSV and fitted PDBs here (default: best-fits)
  --fraglib-config FILE  Location hints (default: ~/.alaric/fraglib.yaml)
  --conda ENV            Run the bash transformation in this conda environment
  -h, --help             Show this help

Fitter options include --exclude-pdb-code CODE, --grid-spacing FLOAT,
--rotamer-precision FLOAT, --ignore-unknown, --ignore-missing,
--ignore-reordered and --no-header. --pdb-output is managed by the wrapper.

Requires Seamless (including --metafile support), SEAMLESS_CACHE or a configured
Seamless cluster, Python 3.12+, numpy<2, scipy, ruamel.yaml, biopython and the fragment
library. See documents/refe-best-fit-seamless.md.
EOF
}

output=best-fits
config="$HOME/.alaric/fraglib.yaml"
environment=()
while (($#)); do
    case "$1" in
        --output|--fraglib-config|--conda)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --output) output=$2 ;;
                --fraglib-config) config=$2 ;;
                --conda) environment=(--conda "$2") ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "Unknown wrapper option: $1" >&2; usage >&2; exit 2 ;;
        *) break ;;
    esac
done
if (($# == 0)); then usage >&2; exit 2; fi
pdb=$1
shift
for argument in "$@"; do
    case "$argument" in
        --pdb-output|--pdb-output=*) echo "Use --output before RNA.pdb" >&2; exit 2 ;;
    esac
done

scriptdir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Keep the caller's working directory for Seamless configuration discovery.
output=$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$output")
inputs_text=$(python -B "$scriptdir/refe_best_fit_seamless_support.py" inputs)
mapfile -t inputs <<< "$inputs_text"
input_args=()
for input in "${inputs[@]}"; do input_args+=(-i "$input"); done
fraglib_identities=$(python -B "$scriptdir/refe_best_fit_seamless_support.py" identities)
seamless-run -y -W -ms "${environment[@]}" \
    "${input_args[@]}" --metafile "$config" \
    --var "fraglib_identities=$fraglib_identities" \
    --metavar "fraglib_config=$(basename -- "$config")" \
    --capture "best-fit-output:$output" \
    python -B "$scriptdir/refe_best_fit_seamless_support.py" run "$pdb" "$@"
