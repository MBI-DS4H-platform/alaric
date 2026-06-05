NCHUNKS={{ nchunks }}
IDX={{ chunk_index }}
chunk_range() {
  local total=$1 idx=$2 n=$3
  local first=$(( (total * (idx - 1)) / n + 1 ))
  local last=$(( (total * idx) / n ))
  echo "$first $last"
}
TOTAL=$({{ python }} - <<"PY"
from library import config
libs, _ = config(verify_checksums=False)
factory = libs["{{ sequence }}"]
lib = factory.create(pdb_code=({{ exclude_python }} or None), only_base=True, with_rotaconformers=False)
print(len(lib.coordinates))
PY
)
read FIRST LAST < <(chunk_range "$TOTAL" "$IDX" "$NCHUNKS")
{{ python }} {{ alaric_dir }}/anchor.py \
  --protein {{ protein_path }} \
  --resid {{ resid }} \
  --sequence {{ sequence }} \
  --dihedral {{ dihedral_args }} \
  --angle {{ angle }} \
  --margin {{ margin }} \
  {{ nucleotide_flag }} \
  --output {{ output_path }} \
  {{ exclude_args }} \
  --bucket-size 16 \
  --conformer-range "$FIRST" "$LAST" \
  --unorganized-subdirs \
  ${ALARIC_ANCHOR_EXTRA_ARGS:-}
### ORGANIZE ###
{{ organize_command }}
