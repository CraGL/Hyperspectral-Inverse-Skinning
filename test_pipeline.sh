#!/usr/bin/env bash

ROOT_DIR="../.."
MODEL_DIR="./models"
TEST_DIR="./results_pipeline"
OBJ_SUFF=".obj"
FLAT_MAX_ITER=10
MVES_MAX_ITER=10

INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_percentile 50 --version 0 --method vertex -rand none"

# OUTPUT_FILE="${SSD_DIR}"/all_error.out

# declare -a gt_models=("cylinder" "cube" "cheburashka" "wolf" "cow")
# declare -a wild_models=("cat-poses" "elephant-gallop" "horse-collapse" "pdance" "samba")
declare -a wild_models=("lion-poses" "chickCrossing" "face-poses" "pcow" "pjump" "crane" "kavanHorse")

# echo -n "" > "${OUTPUT_FILE}"
for name in "${wild_models[@]}"; do
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

		### unmixing hyperspectral per-vertex transformations
		RESULT="${OUTPUT_DIR}/result.txt"
		SIMPLEX_HULL_ARGS="--max-iter ${MVES_MAX_ITER} -O ${RESULT} -R 50"
 		python -u simplex_hull.py "${OUTPUT_DIR}" ${SIMPLEX_HULL_ARGS} 2>&1 | tee -i "${OUTPUT_DIR}"/simplex_hull.out

		## measure differences with ground truth
		python -u ./compare.py "${REST_POSE}" "${MODEL_DIR}"/"${name}" "${MODEL_DIR}"/"${name}"-all/"${name}".DMAT "${OUTPUT}" 2>&1 | tee -a "${OUTPUT_DIR}/recover_poses.out"
		
		## view recovered bones
		cd build
		./viewer2 "../${REST_POSE}" "../${RESULT}"
		cd ..
	done	
done
