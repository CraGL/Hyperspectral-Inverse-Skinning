#!/usr/bin/env bash -x

OUTPUT_DIR=.
ROOT_DIR="../.."

REST_POSE="${ROOT_DIR}"/models/pjump.obj
OTHER_POSE_DIR="${ROOT_DIR}"/models/pjump
# SSD_RESULT="${ROOT_DIR}"/SSD_res/lion-poses-10-output.txt
SSD_RESULT_UNCONSTRAINED="${ROOT_DIR}"/SSD_unconstrained/pjump-20-unconstrained.txt

INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_threshold 1e-4 --version 0"
FLAT_INTERSECTION_ARGS="--strategy ssv:weighted --energy biquadratic --W-projection normalize --x-eps 0.1 --handles 20 --fancy-init ${OUTPUT_DIR}/local_subspace_recover.txt"
SIMPLEX_HULL_ARGS="--method qp-major"

# Generate
mkdir -p "${OUTPUT_DIR}"
python -u "${ROOT_DIR}"/PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${REST_POSE}" "${OTHER_POSE_DIR}"/*.obj -o "${OUTPUT_DIR}"/local_subspace_recover.txt 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover.out
python -u "${ROOT_DIR}"/flat_intersection.py "${REST_POSE}" "${OTHER_POSE_DIR}" ${FLAT_INTERSECTION_ARGS} --output "${OUTPUT_DIR}" 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection.out
# python -u "${ROOT_DIR}"/simplex_hull.py "${OUTPUT_DIR}" ${SIMPLEX_HULL_ARGS} 2>&1 | tee -i "${OUTPUT_DIR}"/simplex_hull.out

# Evaluate
# python -u "${ROOT_DIR}"/recover_poses.py "${REST_POSE}" "${OTHER_POSE_DIR}" "${OUTPUT_DIR}"/result.txt --SSD "${SSD_RESULT}" 2>&1 | tee "${OUTPUT_DIR}"/compare.out
# python -u "${ROOT_DIR}"/recover_poses.py "${REST_POSE}" "${OTHER_POSE_DIR}" "${OUTPUT_DIR}"/result.txt --SSD "${SSD_RESULT_UNCONSTRAINED}" 2>&1 | tee "${OUTPUT_DIR}"/compare_unconstrained.out


# Verify each step

## Verify flat_intersection.py
### All at once:
# parallel python "${ROOT_DIR}"/flat_intersection_apply_output.py "${REST_POSE}" '{}' '{.}.obj' ::: "${OUTPUT_DIR}"/*.DMAT
