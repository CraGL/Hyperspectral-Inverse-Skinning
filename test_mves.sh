#!/usr/bin/env bash

ROOT_DIR="../.."
SSD_DIR="./SSD_unconstrained"
MODEL_DIR="./models"
TEST_DIR="./results_mves"
OBJ_SUFF=".obj"
MAX_ITER=20


# OUTPUT_FILE="${SSD_DIR}"/all_error.out

declare -a gt_models=("cylinder" "cube" "cheburashka" "wolf" "cow")
# declare -a gt_models=("cylinder")

# echo -n "" > "${OUTPUT_FILE}"
for name in "${gt_models[@]}"; do
	REST_POSE="${MODEL_DIR}/${name}.obj"
	for OUTPUT_DIR in "${TEST_DIR}"/"${name}"*
	do
		test_dir=$(basename "${OUTPUT_DIR}")
		H=${test_dir#$name-}
# 		echo python -u simplex_hull.py "${OUTPUT_DIR}" ${SIMPLEX_HULL_ARGS} 2>&1 | tee -i "${OUTPUT_DIR}"/simplex_hull.out
# 		python -u simplex_hull.py "${OUTPUT_DIR}" ${SIMPLEX_HULL_ARGS} 2>&1 | tee -i "${OUTPUT_DIR}"/simplex_hull.out

		## unmixing ground truth per-vertex transformations
		OUTPUT="${OUTPUT_DIR}/result.txt"
		SIMPLEX_HULL_ARGS="--max-iter ${MAX_ITER} -O ${OUTPUT}"
# 		python -u simplex_hull.py "${MODEL_DIR}"/${name} ${SIMPLEX_HULL_ARGS} 2>&1 | tee -i "${OUTPUT_DIR}"/simplex_hull.out

		## measure differences with ground truth
		RECOVER_POSES_OUTPUT="${OUTPUT_DIR}/recover_poses.out"
		python -u ./compare.py "${REST_POSE}" "${MODEL_DIR}"/"${name}" "${MODEL_DIR}"/"${name}"-all/"${name}".DMAT "${OUTPUT}" 2>&1 | tee -a "${RECOVER_POSES_OUTPUT}"
		
		## view recovered bones
		cd build
# 		echo ./viewer2 "../${REST_POSE}" "../${OUTPUT}"
#		./viewer2 "../${REST_POSE}" "../${OUTPUT}"
		cd ..
	done	
done
