#!/usr/bin/env bash

ROOT_DIR="../.."
MODEL_DIR="./models"
RES_DIR="./results-no-init"
OBJ_SUFF=".obj"
MAXITER=10

for REST_POSE in $(basename "${MODEL_DIR}"/cyli*.obj)
do
	name="${REST_POSE%$OBJ_SUFF}"
	POSES_DIR="${MODEL_DIR}/${name}"
	for OUTPUT_DIR in "${RES_DIR}"/"${name}"*
	do
		test_dir=$(basename "${OUTPUT_DIR}")
		H=${test_dir#$name-}
		
		GT_DIR="${MODEL_DIR}/${name}"
		FLAT_INTERSECTION_ARGS_NO_INIT="--energy biquadratic --CSV ${OUTPUT_DIR}/${test_dir}_no_init.csv --max-iter ${MAXITER} --f-eps 0 --handles ${H} --strategy pinv+ssv:weighted --forced-init True"
		echo python3 -u flat_intersection.py "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_NO_INIT} --output "${OUTPUT_DIR}" 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_no_init.out
		python3 -u flat_intersection.py "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_NO_INIT} --output "${OUTPUT_DIR}" 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_no_init.out
		
# 		python -m pdb compare_per_vertex_transformation.py "${GT_DIR}" "${OUTPUT_DIR}"
		
# 		python -u PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}"/*.obj -o "${OUTPUT_DIR}"/local_subspace_recover.txt 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover.out
# 		FLAT_INTERSECTION_ARGS="--energy ipca --CSV ${OUTPUT_DIR}/${test_dir}.csv --W-projection normalize --max-iter ${MAXITER} --f-eps 0 --handles ${H} --fancy-init ${OUTPUT_DIR}/local_subspace_recover.txt"
# 		python -u flat_intersection.py "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection.out
	echo
	done	
done
