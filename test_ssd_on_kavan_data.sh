#!/usr/bin/env bash

SSD_DIR="./SSD_results_on_kavan"
MODEL_DIR="./kavan_models"
OBJ_SUFF=".obj"
OUTPUT_FILE="${SSD_DIR}"/all_error.out

echo -n "" > "${OUTPUT_FILE}"
for rest_pose in $(basename "${MODEL_DIR}"/*.obj)
do
	name="${rest_pose%$OBJ_SUFF}"
	echo "${name}"
	for result in "${SSD_DIR}"/"${name}"*.txt
	do
		python -u ./recover_poses.py "${MODEL_DIR}"/"${rest_pose}" "${MODEL_DIR}"/"${name}" "${result}" 2>&1 | tee -a "${OUTPUT_FILE}"
	done	
done
