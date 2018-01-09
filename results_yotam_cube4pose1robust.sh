#!/usr/bin/env bash -x

OUTPUT_DIR=./results_yotam/cube4pose1_robust

REST_POSE=./models/cube4/cube.obj
OTHER_POSE_DIR=./models/cube4/poses-1

INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_threshold 1e-4 --version 0"
FLAT_INTERSECTION_ARGS="--energy biquadratic -GT "${OTHER_POSE_DIR}" --error True --handles 4 --fancy-init "${OUTPUT_DIR}"/local_subspace_recover.txt"
SIMPLEX_HULL_ARGS="-R 0.01"
# SIMPLEX_HULL_ARGS=

# Generate
mkdir -p "${OUTPUT_DIR}"
python -u PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${REST_POSE}" "${OTHER_POSE_DIR}"/*.obj -o "${OUTPUT_DIR}"/local_subspace_recover.txt 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover.out
python -u flat_intersection.py "${REST_POSE}" "${OTHER_POSE_DIR}" ${FLAT_INTERSECTION_ARGS} --output "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}"/flat_intersection.out
python3 -u simplex_hull.py "${OUTPUT_DIR}" ${SIMPLEX_HULL_ARGS} 2>&1 | tee "${OUTPUT_DIR}"/simplex_hull.out

# Evaluate
python -u compare.py "${REST_POSE}" "${OTHER_POSE_DIR}" "$(dirname "${REST_POSE}")"/"$(basename "${REST_POSE}" .obj)".DMAT "${OUTPUT_DIR}"/result.txt 2>&1 | tee "${OUTPUT_DIR}"/compare.out

# Verify each step

## Verify flat_intersection.py
### All at once:
parallel python flat_intersection_apply_output.py "${REST_POSE}" '{}' '{.}.obj' ::: "${OUTPUT_DIR}"/*.DMAT
