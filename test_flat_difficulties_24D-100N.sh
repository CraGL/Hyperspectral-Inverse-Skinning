#!/usr/bin/env bash

DIM=24
NUM=100
# let "MAX_ITER = $DIM - 1"
OUTPUT_DIR=test_difficulties

> "${OUTPUT_DIR}"/test_flat_difficulites.out-"${DIM}"D-"${NUM}"N
for q in `seq 1 ${DIM}`; do
	for h in `seq 1 ${DIM}`; do
		echo $q $h
		python3 -u pymanopt_test_karcher.py --manifold pB --optimize-from random --number ${NUM} --dim ${DIM} --ortho ${q} --handles ${h} --test handle --pinv true 2>&1 | tee -a -i "${OUTPUT_DIR}"/test_flat_difficulites.out
	done
done
    
