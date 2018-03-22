#!/usr/bin/env bash

ROOT_DIR="../.."
MODEL_DIR="./models"
RES_DIR="./results_strategy"
OBJ_SUFF=".obj"
MAXITER=20

POSES_DIR="${MODEL_DIR}/${name}"
for OUTPUT_DIR in "${RES_DIR}"/cat*
do
	test_dir=$(basename "${OUTPUT_DIR}")
	SUF=`expr "$test_dir" : '.*\(-[0-9]*\)'` 
	H="${SUF:1}"
	name="${test_dir%$SUF}"
	REST_POSE="${MODEL_DIR}"/"${name}".obj
	POSES_DIR="${MODEL_DIR}"/"${name}"
	H=${test_dir#$name-}
	for SEED in `seq 0 0`
	do
# 		FLAT_INTERSECTION_ARGS_BIQUADRATIC="--energy biquadratic --forced-init True --subset 100 --CSV ${OUTPUT_DIR}/${test_dir}_biquadratic_100.csv --strategy pinv+ssv:weighted --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
# 		python -m pdb flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_BIQUADRATIC} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_biquadratic.out

# 		FLAT_INTERSECTION_ARGS_BIQUADRATIC="--energy biquadratic --forced-init True --seed $SEED --CSV ${OUTPUT_DIR}/${test_dir}_sparse4_${SEED}.csv --strategy pinv+ssv:weighted --z-strategy sparse4 --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
# 		python -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_BIQUADRATIC} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_biquadratic.out

		FLAT_INTERSECTION_ARGS_BIQUADRATIC="--energy biquadratic --forced-init True --seed $SEED --CSV ${OUTPUT_DIR}/${test_dir}_basinhopping4.csv --strategy pinv+ssv:weighted --basinhopping 4 --max-iter 5 --f-eps 0 --handles ${H}"
		python -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_BIQUADRATIC} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_biquadratic.out


# 		FLAT_INTERSECTION_ARGS_BIQUADRATIC="--energy biquadratic --forced-init True --seed $SEED --CSV ${OUTPUT_DIR}/${test_dir}_biquadratic_${SEED}.csv --strategy pinv+ssv:weighted --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
# 		python -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_BIQUADRATIC} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_biquadratic.out
# 
# 		FLAT_INTERSECTION_ARGS_B="--energy B --forced-init True --seed $SEED --CSV ${OUTPUT_DIR}/${test_dir}_b_${SEED}.csv --W-projection normalize --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
# 		python -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_B} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_b.out
# 		
# 		FLAT_INTERSECTION_ARGS_CAYLEY="--energy cayley --seed $SEED --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_cayley_${SEED}.csv --W-projection normalize --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
# 		python -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_CAYLEY} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_cayley.out
# 
# 		FLAT_INTERSECTION_ARGS_IPCA="--energy ipca --seed $SEED --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_ipca_${SEED}.csv --W-projection normalize --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
# 		python -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_IPCA} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_cayley.out	
	done
echo
done	
