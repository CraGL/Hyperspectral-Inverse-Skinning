#!/usr/bin/env bash

ROOT_DIR="../.."
MODEL_DIR="./models"
RES_DIR="./results_strategy_compare_flat_method_without_initial_guess"
OBJ_SUFF=".obj"
MAXITER=20



POSES_DIR="${MODEL_DIR}/${name}"
# for OUTPUT_DIR in "${RES_DIR}"/cat-poses*  "${RES_DIR}"/elephant-gallop* "${RES_DIR}"/horse-collapse*  "${RES_DIR}"/chickenCrossing*  "${RES_DIR}"/pdance*
# for OUTPUT_DIR in "${RES_DIR}"/horse-collapse*  "${RES_DIR}"/chickenCrossing*  "${RES_DIR}"/pdance*

# for OUTPUT_DIR in "${RES_DIR}"/cylinder*  "${RES_DIR}"/cube* "${RES_DIR}"/cheburashka*  "${RES_DIR}"/wolf*  "${RES_DIR}"/cow*
for OUTPUT_DIR in "${RES_DIR}"/cat-poses*  "${RES_DIR}"/cheburashka*
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
		## biquadratic
		FLAT_INTERSECTION_ARGS_BIQUADRATIC="--energy biquadratic --forced-init True --seed $SEED --CSV ${OUTPUT_DIR}/${test_dir}_biquadratic_${SEED}.csv --strategy pinv+ssv:weighted --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
		python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_BIQUADRATIC} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_biquadratic.out

		## B
		FLAT_INTERSECTION_ARGS_B="--energy B --forced-init True --seed $SEED --CSV ${OUTPUT_DIR}/${test_dir}_b_${SEED}.csv --W-projection normalize --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
		python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_B} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_b.out
 				
		## IPCA 
		FLAT_INTERSECTION_ARGS_IPCA="--energy ipca --seed $SEED --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_ipca_${SEED}.csv --W-projection normalize --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
		python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_IPCA} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_ipca.out	

        # # ## pymanopt pB
        FLAT_INTERSECTION_ARGS_BIQUADRATIC="--energy B --forced-init True --seed $SEED --max-iter 0 --f-eps 0 --handles ${H}"
        python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_BIQUADRATIC} --save-matlab-initial "${OUTPUT_DIR}"/filename.mat
        x=$((H-1))
        echo $x
        python3 -u pymanopt_test_karcher.py --load "${OUTPUT_DIR}"/filename.mat --test-data "${OUTPUT_DIR}"/filename.mat --handles $x --optimize-solver conjugate --manifold pB  2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_pymanopt_pB.out

	done
echo
done	
