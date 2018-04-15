#!/usr/bin/env bash

ROOT_DIR="../.."
SSD_DIR="./SSD_unconstrained"
MODEL_DIR="./models"
# TEST_DIR="./results_songrun"
TEST_DIR="./results_clean_vertex_0_use_absolute_initial_guess_threshold"
OBJ_SUFF=".obj"
MAXITER=10

OUTPUT_FILE="${SSD_DIR}"/all_error.out

INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_threshold 1e-4 --version 0 --method vertex"

declare -a wild_models=("cat-poses" "elephant-gallop" "horse-collapse" "chickenCrossing" "pdance")
declare -a gt_models=("cube" "cylinder" "cheburashka" "wolf" "cow")


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
		
		# FLAT_INTERSECTION_ARGS="--energy biquadratic --max-iter ${MAXITER} --handles ${H} --f-eps 0 --strategy pinv+ssv:weighted --forced-init True"
		# python3 -u flat_intersection.py "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection.out
		
		for initial_method in "none"; do	
 	
			
			### not save dmat, because conflict with flat_intersection saved dmat
			python -u PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}"/*.obj -rand $initial_method -o "${OUTPUT_DIR}"/local_subspace_recover_${initial_method}.txt 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover-${initial_method}.out

            ### save dmat for initial guess and save dmat for flat_intersection.
			# python2 -u PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}"/*.obj -rand $initial_method -o "${OUTPUT_DIR}"/local_subspace_recover_${initial_method}.txt --save-dmat "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover-${initial_method}.out
			### generate deformed meshes from per-vertex transformation
			# python3 generate_deformed_meshes.py "${MODEL_DIR}"/"${REST_POSE}" "${MODEL_DIR}"/"${name}-all"/"${name}.DMAT" "${OUTPUT_DIR}"
			

			## flat intersection, can see vertex error in .out files.
			FLAT_INTERSECTION_ARGS="--energy biquadratic --max-iter ${MAXITER} --handles ${H} --f-eps 0 --strategy pinv+ssv:weighted --forced-init True -I ${OUTPUT_DIR}/local_subspace_recover_${initial_method}.txt"
			python3 -u flat_intersection.py "${MODEL_DIR}"/"${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS} --output "${OUTPUT_DIR}"/"${initial_method}" 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection-${initial_method}.out
		   
		   	### generate deformed meshes from per-vertex transformation
			python3 generate_deformed_meshes.py "${MODEL_DIR}"/"${REST_POSE}" "${MODEL_DIR}"/"${name}-all"/"${name}.DMAT" "${OUTPUT_DIR}"/"${initial_method}"

		    #### compute per vertex errors and saved in .out files, only for gt_models 5 examples.
            python3 -u compare_per_vertex_transformation.py "${GT_DIR}" "${OUTPUT_DIR}"/"${initial_method}" 2>&1 | tee -i "${OUTPUT_DIR}"/compare_per_vertex_transformation-${initial_method}.out


		done
	done	
done
