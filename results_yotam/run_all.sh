#!/usr/bin/env bash

for dir in cheburashka_pose1_default cheburashka_pose4_default cube4pose1_default cube4pose1_flat_intersection_ground_truth_output cube4pose1_flat_thresh cube4pose1_simplex_robust cube4pose1_stricter_guess
do
    echo ==== Entering "${dir}" ====
    (cd "${dir}" && ./run.sh)
done
