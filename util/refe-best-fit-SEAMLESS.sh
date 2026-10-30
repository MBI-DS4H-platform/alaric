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

Required fitter input: --exclude-pdb-code CODE. The only optional fitter
argument is --rotamer-precision 0.5.
Grid spacing is fixed at sqrt(3)/3. --no-header and ignore flags are forbidden;
--pdb-output is managed by the wrapper.

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
exec python -B "$scriptdir/refe-best-fit-SEAMLESS.py" \
    --output "$output" --fraglib-config "$config" \
    "${environment[@]}" "$pdb" "$@"
