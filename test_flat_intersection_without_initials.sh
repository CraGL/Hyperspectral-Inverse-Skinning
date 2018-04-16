#!/usr/bin/env bash

ROOT_DIR="../.."
SSD_DIR="./SSD_unconstrained"
MODEL_DIR="./models"
# TEST_DIR="./results_songrun"
TEST_DIR="./results_clean_without_initials"
OBJ_SUFF=".obj"
MAXITER=10

OUTPUT_FILE="${SSD_DIR}"/all_error.out

declare -a wild_models=("cat-poses" "elephant-gallop" "horse-collapse" "lion-poses" "pdance")
declare -a gt_models=("cube" "cylinder" "cheburashka" "wolf" "cow")
# declare -a gt_models=("cylinder")


echo -n "" > "${OUTPUT_FILE}"
# for REST_POSE in $(basename "${MODEL_DIR}"/cyli*.obj)
for name in "${gt_models[@]}"; do
	REST_POSE="${name}.obj"
	POSES_DIR="${MODEL_DIR}/${name}"
	for OUTPUT_DIR in "${TEST_DIR}"/"${name}"*
	do
		test_dir=$(basename "${OUTPUT_DIR}")
		H=${test_dir#$name-}
		
		GT_DIR="${MODEL_DIR}/${name}"
		
		## flat intersection, can see vertex error in .out files.
		FLAT_INTERSECTION_ARGS="--energy biquadratic --max-iter ${MAXITER} --handles ${H} --f-eps 0 --strategy pinv+ssv:weighted --forced-init True"
		python3 -u flat_intersection.py "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS} --output "${OUTPUT_DIR}" 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection.out
	   
	   	### generate deformed meshes from per-vertex transformation
		python3 generate_deformed_meshes.py "${MODEL_DIR}"/"${REST_POSE}" "${MODEL_DIR}"/"${name}-all"/"${name}.DMAT" "${OUTPUT_DIR}"

	    #### compute per vertex errors and saved in .out files, only for gt_models 5 examples.
        python3 -u compare_per_vertex_transformation.py "${GT_DIR}" "${OUTPUT_DIR}" 2>&1 | tee -i "${OUTPUT_DIR}"/compare_per_vertex_transformation.out

	done	
done
