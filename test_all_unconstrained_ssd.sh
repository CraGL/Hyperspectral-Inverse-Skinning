#!/usr/bin/env bash

SSD_DIR="./SSD_unconstrained"
MODEL_DIR="./models"
OBJ_SUFF=".obj"
OUTPUT_FILE="${SSD_DIR}"/all_error.out

echo -n "" > "${OUTPUT_FILE}"
for rest_pose in $(basename "${MODEL_DIR}"/cat*.obj)
do
	name="${rest_pose%$OBJ_SUFF}"
	for result in "${SSD_DIR}"/"${name}"*.txt
	do
		python -u ./recover_poses.py --output NO "${MODEL_DIR}"/"${rest_pose}" "${MODEL_DIR}"/"${name}" "${result}" 2>&1 | tee -a "${OUTPUT_FILE}"
	done	
done
