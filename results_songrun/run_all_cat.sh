#!/usr/bin/env bash

for dir in cat_pose_Wnorm_handle10 cat_pose_Wnorm_handle15 cat_pose_Wnorm_handle20 cat_pose_Wnorm_handle25
do
    echo ==== Entering "${dir}" ====
    (cd "${dir}" && ./run.sh)
done
