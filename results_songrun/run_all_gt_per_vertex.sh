#!/usr/bin/env bash

for dir in cheburashka_pose1_default cheburashka_pose2_default cube_pose1_default cube_pose2_default wolf_pose1_default wolf_pose2_default cow_pose1_default cow_pose2_default
do
    echo ==== Entering "${dir}" ====
    (cd "${dir}" && ./run.sh)
done
