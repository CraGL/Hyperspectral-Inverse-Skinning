#!/usr/bin/env bash

ROOT_DIR="../.."
MODEL_DIR="./models"
TEST_DIR="./results_clean"
OBJ_SUFF=".obj"
FLAT_MAX_ITER=4

INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_percentile 50 --version 0 --method vertex -rand none"
declare -a gt_models=("cylinder" "cube" "cheburashka" "wolf" "cow")
declare -a wild_models=("cat-poses" "elephant-gallop" "elephant-poses" "chickenCrossing" "face-poses" "horse-collapse" "horse-gallop" "horse-poses" "lion-poses" "pcow" "pdance" "pjump")
declare -a kavan_models=("crane" "elasticCow" "kavanElephant" "kavanHorse" "samba")

# for REST_POSE in $(basename "${MODEL_DIR}"/cyli*.obj)
for name in "${gt_models[@]}"; do
	REST_POSE="${MODEL_DIR}/${name}.obj"
	POSES_DIR="${MODEL_DIR}/${name}"
	for OUTPUT_DIR in "${TEST_DIR}"/"${name}"*
	do
		test_dir=$(basename "${OUTPUT_DIR}")
		H=${test_dir#$name-}
		
		### initial guess
		python -u PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${REST_POSE}" "${POSES_DIR}"/*.obj -o "${OUTPUT_DIR}"/local_subspace_recover.txt 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover.out

		### flat optimization
		FLAT_INTERSECTION_ARGS="--energy biquadratic --max-iter ${FLAT_MAX_ITER} --handles ${H} --f-eps 0 --strategy pinv+ssv:weighted --forced-init True -I ${OUTPUT_DIR}/local_subspace_recover.txt --output ${OUTPUT_DIR}"
		python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection.out

		### generate deformed meshes from per-vertex transformation
# 		python generate_deformed_meshes.py "${REST_POSE}" "${MODEL_DIR}"/"${name}-all"/"${name}.DMAT" "${OUTPUT_DIR}"
		
		
	done	
done
