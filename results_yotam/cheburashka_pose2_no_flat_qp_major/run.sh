#!/usr/bin/env bash -x

OUTPUT_DIR=.
ROOT_DIR="../.."

REST_POSE="${ROOT_DIR}"/models/cheburashka/cheburashka.obj
OTHER_POSE_DIR="${ROOT_DIR}"/models/cheburashka/poses-2
SSD_RESULT="${ROOT_DIR}"/SSD_res/cheb11-2-output.txt

INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_threshold 1e-4 --version 0"
FLAT_INTERSECTION_ARGS="--energy biquadratic -GT "${OTHER_POSE_DIR}" --error True --handles 11 --fancy-init "${OUTPUT_DIR}"/local_subspace_recover.txt"
SIMPLEX_HULL_ARGS="-D 10 --method qp-major"

# Generate
mkdir -p "${OUTPUT_DIR}"
python -u "${ROOT_DIR}"/PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${REST_POSE}" "${OTHER_POSE_DIR}"/*.obj -o "${OUTPUT_DIR}"/local_subspace_recover.txt 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover.out
python3 -u "${ROOT_DIR}"/simplex_hull_with_initial_guess.py "${OUTPUT_DIR}"/local_subspace_recover.txt "${REST_POSE}" "${OTHER_POSE_DIR}" "${OUTPUT_DIR}"/result.txt ${SIMPLEX_HULL_ARGS} 2>&1 | tee -i "${OUTPUT_DIR}"/simplex_hull.out

# Evaluate
python -u "${ROOT_DIR}"/compare.py --write-OBJ True --SSD "${SSD_RESULT}" "${REST_POSE}" "${OTHER_POSE_DIR}" "$(dirname "${REST_POSE}")"/"$(basename "${REST_POSE}" .obj)".DMAT "${OUTPUT_DIR}"/result.txt 2>&1 | tee "${OUTPUT_DIR}"/compare.out

# Verify each step

## Verify flat_intersection.py
### All at once:
# parallel python "${ROOT_DIR}"/flat_intersection_apply_output.py "${REST_POSE}" '{}' '{.}.obj' ::: "${OUTPUT_DIR}"/*.DMAT