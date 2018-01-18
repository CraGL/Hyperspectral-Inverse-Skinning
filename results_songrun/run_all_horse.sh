#!/usr/bin/env bash

for dir in horse_collapse_default_handle10 horse_collapse_default_handle20 horse_gallop_default_handle10 horse_gallop_default_handle20 horse_gallop_default_handle30 horse_poses_default_handle10 horse_poses_default_handle20 
do
    echo ==== Entering "${dir}" ====
    (cd "${dir}" && ./run.sh)
done
