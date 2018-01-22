#!/usr/bin/env bash

ROOT_DIR="../.."
KAVAN_DIR="./kavan_res"
MODEL_DIR="./kavan_models"
TEST_DIR="./results_songrun"
OBJ_SUFF=".obj"

OUTPUT_FILE="${KAVAN_DIR}"/all_error.out

INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_threshold 1e-4 --version 0"

echo -n "" > "${OUTPUT_FILE}"
for REST_POSE in $(basename "${MODEL_DIR}"/*.obj)
do
	name="${REST_POSE%$OBJ_SUFF}"
	POSES_DIR="${MODEL_DIR}/${name}"
	for OUTPUT_DIR in "${TEST_DIR}"/"${name}"-[0-9][0-9]
	do
		test_dir=$(basename "${OUTPUT_DIR}")
		H=${test_dir#$name-}
# 		FLAT_INTERSECTION_ARGS="--energy biquadratic --W-projection normalize --max-iter 2 --handles ${H} --fancy-init ${OUTPUT_DIR}/local_subspace_recover.txt"
# 		python -u PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}"/*.obj -o "${OUTPUT_DIR}"/local_subspace_recover.txt 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover.out
		
		FLAT_INTERSECTION_ARGS="--energy biquadratic --W-projection normalize --max-iter 2 --handles ${H}"
		python -u flat_intersection.py "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS} --output "${OUTPUT_DIR}" 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection.out
	echo
	done	
done
