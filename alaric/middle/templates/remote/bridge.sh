export TMPDIR="${ALARIC_REMOTE_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
{{ python }} {{ alaric_dir }}/bridge.py \
  --lower-poses {{ lower_result_path }} \
  --upper-poses {{ upper_result_path }} \
  --lower-sequence {{ lower_sequence }} \
  --middle-sequence {{ middle_sequence }} \
  --upper-sequence {{ upper_sequence }} \
  --lower-crmsd {{ lower_crmsd }} \
  --lower-ov-rmsd {{ lower_ovrmsd }} \
  --upper-crmsd {{ upper_crmsd }} \
  --upper-ov-rmsd {{ upper_ovrmsd }} \
  --output {{ output_path }} \
  --memory {{ memory }} \
  --max-intermediate-poses {{ max_intermediate_poses }} \
  --max-final-poses {{ max_final_poses }} \
  --nprocs {{ nprocs }} \
  --rotamer-chunks {{ rotamer_chunks }} \
  --estimator-seed {{ estimator_seed }}
