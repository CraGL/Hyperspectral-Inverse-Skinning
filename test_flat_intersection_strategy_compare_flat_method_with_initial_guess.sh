#!/usr/bin/env bash

ROOT_DIR="../.."
MODEL_DIR="./models"
RES_DIR="./results_strategy_compare_flat_method_with_initial_guess"
OBJ_SUFF=".obj"
MAXITER=20

INITIAL_GUESS_ARGS="--svd_threshold 1e-15 --transformation_percentile 50 --version 0 --method vertex -rand none"


POSES_DIR="${MODEL_DIR}/${name}"
# for OUTPUT_DIR in "${RES_DIR}"/cat-poses*  "${RES_DIR}"/elephant-gallop* "${RES_DIR}"/horse-collapse*  "${RES_DIR}"/chickenCrossing*  "${RES_DIR}"/pdance*
# for OUTPUT_DIR in "${RES_DIR}"/horse-collapse*  "${RES_DIR}"/chickenCrossing*  "${RES_DIR}"/pdance*

# for OUTPUT_DIR in "${RES_DIR}"/cylinder*  "${RES_DIR}"/cube* "${RES_DIR}"/cheburashka*  "${RES_DIR}"/wolf*  "${RES_DIR}"/cow*
# for OUTPUT_DIR in "${RES_DIR}"/cheburashka* "${RES_DIR}"/cat-poses* 
for OUTPUT_DIR in "${RES_DIR}"/cylinder*
# for OUTPUT_DIR in "${RES_DIR}"/cheburashka* 
do
	test_dir=$(basename "${OUTPUT_DIR}")
	SUF=`expr "$test_dir" : '.*\(-[0-9]*\)'` 
	H="${SUF:1}"
	name="${test_dir%$SUF}"
	REST_POSE="${MODEL_DIR}"/"${name}".obj
	POSES_DIR="${MODEL_DIR}"/"${name}"
	H=${test_dir#$name-}

    #### if print-all is True, then all qs_data are used for flat_intersection.py
	python -u PerVertex/local_subspace_recover.py ${INITIAL_GUESS_ARGS} "${REST_POSE}" "${POSES_DIR}"/*.obj --print-all True --out "${OUTPUT_DIR}"/local_subspace_recover.txt --out-errors "${OUTPUT_DIR}"/local_subspace_recover_errors.txt --out-ssv "${OUTPUT_DIR}"/local_subspace_recover_ssv.txt  2>&1 | tee "${OUTPUT_DIR}"/local_subspace_recover.out

	## biquadratic
	FLAT_INTERSECTION_ARGS_BIQUADRATIC="--energy biquadratic --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_biquadratic.csv --strategy pinv+ssv:weighted --max-iter ${MAXITER} --f-eps 0 --handles ${H} -I ${OUTPUT_DIR}/local_subspace_recover.txt"
	python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_BIQUADRATIC} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_biquadratic.out

	## B
	FLAT_INTERSECTION_ARGS_B="--energy B --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_b.csv --W-projection normalize --max-iter ${MAXITER} --f-eps 0 --handles ${H} -I ${OUTPUT_DIR}/local_subspace_recover.txt"
	python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_B} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_b.out

	## IPCA 
	FLAT_INTERSECTION_ARGS_IPCA="--energy ipca --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_ipca.csv --W-projection normalize --max-iter ${MAXITER} --f-eps 0 --handles ${H} -I ${OUTPUT_DIR}/local_subspace_recover.txt"
	python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_IPCA} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_ipca.out	

	## pb_pymanopt conjugate
	FLAT_INTERSECTION_ARGS_PB_PYMANOPT_CONJUGATE="--energy pB_pymanopt --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_pB_pymanopt_conjugate.csv --strategy conjugate --max-iter ${MAXITER} --f-eps 0 --handles ${H} -I ${OUTPUT_DIR}/local_subspace_recover.txt"
	python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_PB_PYMANOPT_CONJUGATE} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_pB_pymanopt_conjugate.out

	## pb_pymanopt trust
	FLAT_INTERSECTION_ARGS_PB_PYMANOPT_TRUST="--energy pB_pymanopt --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_pB_pymanopt_trust.csv --strategy trust --max-iter ${MAXITER} --f-eps 0 --handles ${H} -I ${OUTPUT_DIR}/local_subspace_recover.txt"
	python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_PB_PYMANOPT_TRUST} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_pB_pymanopt_trust.out

	## pb_pymanopt steepest
	FLAT_INTERSECTION_ARGS_PB_PYMANOPT_STEEPEST="--energy pB_pymanopt --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_pB_pymanopt_steepest.csv --strategy steepest --max-iter ${MAXITER} --f-eps 0 --handles ${H} -I ${OUTPUT_DIR}/local_subspace_recover.txt"
	python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_PB_PYMANOPT_STEEPEST} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_pB_pymanopt_steepest.out

	## intersection conjugate
	FLAT_INTERSECTION_ARGS_INTERSECTION="--energy intersection --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_intersection_conjugate.csv --strategy conjugate --max-iter ${MAXITER} --f-eps 0 --handles ${H} -I ${OUTPUT_DIR}/local_subspace_recover.txt"
	python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_INTERSECTION} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_intersection_conjugate.out

	# ## laplacian
	FLAT_INTERSECTION_ARGS_LAPLACIAN="--energy laplacian --forced-init True --CSV ${OUTPUT_DIR}/${test_dir}_laplacian.csv --max-iter ${MAXITER} --f-eps 0 --handles ${H} --fancy-init ${OUTPUT_DIR}/local_subspace_recover.txt --fancy-init-errors ${OUTPUT_DIR}/local_subspace_recover_errors.txt --fancy-init-ssv ${OUTPUT_DIR}/local_subspace_recover_ssv.txt"
	python3 -u flat_intersection.py "${REST_POSE}" "${POSES_DIR}" ${FLAT_INTERSECTION_ARGS_LAPLACIAN} 2>&1 | tee -i "${OUTPUT_DIR}"/flat_intersection_laplacian.out

echo
done	
