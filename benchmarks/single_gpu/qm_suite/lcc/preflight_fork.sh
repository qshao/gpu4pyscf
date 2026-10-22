#!/bin/bash
# usage: preflight_fork.sh FORK_DIR CASE_LIST_FILE REPS -- verifies on the login node that schedule.py accepts the fork (cases, policy, source identities).
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
python schedule.py --campaign all --output $1 --cases $(cat $2) --repetitions $3 --references-only --gpu-hours 1 --cpu-hours 1000 2>&1 | tail -3
