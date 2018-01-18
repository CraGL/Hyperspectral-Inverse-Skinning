#!/usr/bin/env bash

for dir in elephant_gallop_default_handle10 elephant_gallop_default_handle20 elephant_gallop_default_handle27 elephant_poses_default_handle10 elephant_poses_default_handle21 
do
    echo ==== Entering "${dir}" ====
    (cd "${dir}" && ./run.sh)
done
