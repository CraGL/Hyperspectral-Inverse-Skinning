#!/usr/bin/env bash

RES_DIR="./kavan_res"
MODEL_DIR="./kavan_models"
RES_SUFF=".skin.txt"
OUTPUT_FILE="${RES_DIR}"/all_error.out

echo -n "" > "${OUTPUT_FILE}"
for res_file in $(basename "${RES_DIR}"/elasticCow.skin.txt)
do
	name="${res_file%$RES_SUFF}"
	echo "${name}"
# 	if test "$name" = 'elasticCow'; then
# 		name="pcow"
# 	elif test "$name" = 'elephant'; then
# 		name="elephant-gallop"
# 	elif test "$name" = 'horse'; then
# 		name="horse-gallop"
# 	fi
	python -u ./recover_poses.py --kavan True --showAll True "${RES_DIR}"/"${name}".obj "${MODEL_DIR}"/"${name}" "./kavan_res"/"${res_file}" 2>&1 | tee -a "${OUTPUT_FILE}"
		
done
