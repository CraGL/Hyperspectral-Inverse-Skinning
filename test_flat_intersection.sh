#!/usr/bin/env bash

ROOT_DIR="../.."
SSD_DIR="./SSD_unconstrained"
MODEL_DIR="./models"
TEST_DIR="./results_songrun"
OBJ_SUFF=".obj"

OUTPUT_FILE="${SSD_DIR}"/all_error.out

INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_threshold 1e-4 --version 0"

echo -n "" > "${OUTPUT_FILE}"
for REST_POSE in $(basename "${MODEL_DIR}"/cheb*.obj)
do
	name="${REST_POSE%$OBJ_SUFF}"
	POSES_DIR="${MODEL_DIR}/${name}"
	for OUTPUT_DIR in "${TEST_DIR}"/"${name}"*
	do
		test_dir=$(basename "${OUTPUT_DIR}")
		H=${test_dir#$name-}
		
		GT_DIR="${MODEL_DIR}/${name}"
		FLAT_INTERSECTION_ARGS="--energy biquadratic --max-iter 2 --handles ${H} --fancy-init ${OUTPUT_DIR}/local_subspace_recover.txt --strategy pinv+ssv:weighted"
# 		FLAT_INTERSECTION_ARGS="--energy biquadratic --W-projection normalize --x-eps 0.1 --f-eps 0 --handles ${H} --fancy-init ${OUTPUT_DIR}/local_subspace_recover.txt"
		echo python -u PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}"/*.obj -o "${OUTPUT_DIR}"/local_subspace_recover.txt 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover.out
		python -u flat_intersection.py "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS} --output "${OUTPUT_DIR}" 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection.out
		
		python -u compare_per_vertex_transformation.py "${GT_DIR}" "${OUTPUT_DIR}"
	echo
	done	
done
