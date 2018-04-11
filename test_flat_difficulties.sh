#!/usr/bin/env bash

DIM=12
NUM=20
# let "MAX_ITER = $DIM - 1"
OUTPUT_DIR=test_difficulties

> "${OUTPUT_DIR}"/test_flat_difficulites.out
for q in `seq 1 ${DIM}`; do
	for h in `seq 1 ${DIM}`; do
		echo $q $h
		python3 -u pymanopt_test_karcher.py --manifold pB --optimize-from random --number ${NUM} --dim ${DIM} --ortho ${q} --handles ${h} --test handle 2>&1 | tee -a -i "${OUTPUT_DIR}"/test_flat_difficulites.out
	done
done
    
