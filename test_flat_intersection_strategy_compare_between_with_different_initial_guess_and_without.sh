#!/usr/bin/env bash

ROOT_DIR="../.."
MODEL_DIR="./models"
RES_DIR="./results_strategy_compare_between_with_different_initial_guess_and_without"
OBJ_SUFF=".obj"
MAXITER=20



POSES_DIR="${MODEL_DIR}/${name}"
# for OUTPUT_DIR in "${RES_DIR}"/cylinder*  "${RES_DIR}"/cube* "${RES_DIR}"/cheburashka*  "${RES_DIR}"/wolf*  "${RES_DIR}"/cow*  "${RES_DIR}"/cat-poses*  "${RES_DIR}"/elephant-gallop* "${RES_DIR}"/horse-collapse*  "${RES_DIR}"/chickenCrossing*  "${RES_DIR}"/pdance*
# for OUTPUT_DIR in "${RES_DIR}"/cow*  "${RES_DIR}"/cat-poses*  "${RES_DIR}"/horse-collapse*  "${RES_DIR}"/chickenCrossing*  "${RES_DIR}"/pdance*
for OUTPUT_DIR in "${RES_DIR}"/cheburashka*  "${RES_DIR}"/cat-poses*
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
	    for initial_method in "none" "euclidian" "geodesic" 
	    do 	
	    	for VERSION in `seq 0 1`
	    	do
				INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_percentile 10 --version ${VERSION} --method vertex"
				python -u PerVertex/local_subspace_recover.py "${REST_POSE}" "${POSES_DIR}"/*.obj ${INITIAL_GUESS_ARGS} -rand $initial_method -o "${OUTPUT_DIR}"/local_subspace_recover-${initial_method}_version_${VERSION}.txt
				
				FLAT_INTERSECTION_ARGS_BIQUADRATIC="--energy biquadratic --forced-init True --seed $SEED --CSV ${OUTPUT_DIR}/${test_dir}_biquadratic_${SEED}-${initial_method}_version_${VERSION}.csv --strategy pinv+ssv:weighted --max-iter ${MAXITER} --f-eps 0 --handles ${H} -I ${OUTPUT_DIR}/local_subspace_recover-${initial_method}_version_${VERSION}.txt"
			    python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_BIQUADRATIC}  2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection-${initial_method}_version_${VERSION}.out

			done
		done
        
        ### no initial guess
		FLAT_INTERSECTION_ARGS_BIQUADRATIC="--energy biquadratic --forced-init True --seed $SEED --CSV ${OUTPUT_DIR}/${test_dir}_biquadratic_${SEED}-without_initial_guess.csv --strategy pinv+ssv:weighted --max-iter ${MAXITER} --f-eps 0 --handles ${H}"
		python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_BIQUADRATIC}  2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection-without_initial_guess.out

	done
done	
