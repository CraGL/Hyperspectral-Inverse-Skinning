#!/usr/bin/env bash -x

OUTPUT_DIR=.
ROOT_DIR="../.."

REST_POSE="${ROOT_DIR}"/models/cube.obj
OTHER_POSE_DIR="${ROOT_DIR}"/models/cube

INITIAL_GUESS_ARGS="--version 0 --print-all True"
FLAT_INTERSECTION_ARGS="--energy laplacian --f-eps 0 --x-eps 0.01 -GT "${OTHER_POSE_DIR}" --error True --handles 4 --fancy-init "${OUTPUT_DIR}"/local_subspace_recover.txt --fancy-init-errors "${OUTPUT_DIR}"/local_subspace_recover_errors.txt --fancy-init-ssv "${OUTPUT_DIR}"/local_subspace_recover_ssv.txt"
# SIMPLEX_HULL_ARGS="-R 0.01"
SIMPLEX_HULL_ARGS=

# Generate
mkdir -p "${OUTPUT_DIR}"
# python -u "${ROOT_DIR}"/PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${REST_POSE}" "${OTHER_POSE_DIR}"/*.obj --out "${OUTPUT_DIR}"/local_subspace_recover.txt --out-errors "${OUTPUT_DIR}"/local_subspace_recover_errors.txt --out-ssv "${OUTPUT_DIR}"/local_subspace_recover_ssv.txt 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover.out
python -u "${ROOT_DIR}"/flat_intersection.py "${REST_POSE}" "${OTHER_POSE_DIR}" ${FLAT_INTERSECTION_ARGS} --output "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}"/flat_intersection.out
python3 -u "${ROOT_DIR}"/simplex_hull.py "${OUTPUT_DIR}" ${SIMPLEX_HULL_ARGS} 2>&1 | tee "${OUTPUT_DIR}"/simplex_hull.out

# Evaluate
python -u "${ROOT_DIR}"/compare.py "${REST_POSE}" "${OTHER_POSE_DIR}" "$(dirname "${REST_POSE}")"/"$(basename "${REST_POSE}" .obj)".DMAT "${OUTPUT_DIR}"/result.txt 2>&1 | tee "${OUTPUT_DIR}"/compare.out

# Verify each step

## Verify flat_intersection.py
### All at once:
parallel python "${ROOT_DIR}"/flat_intersection_apply_output.py "${REST_POSE}" '{}' '{.}.obj' ::: "${OUTPUT_DIR}"/*.DMAT
