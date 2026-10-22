#!/bin/bash
# usage: launch_engine.sh MASTER_NAME CASE_LIST REPS CARD...   -- the CuPy-fallback engine stratum (cuTENSOR off) of an existing campaign
set -e
MASTER=$1; CASES=$2; REPS=$3; shift 3
S=/scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
CACHE=/scratch/qsh226/g4p_lcc/cache
for card in "$@"; do
  case $card in
    a100) PART=A2V80_ICE56M256_L;;
    v100) PART=V4V32_SKY32M192_L;;
    *) echo "engine stratum is for cards that have cuTENSOR: a100 v100"; exit 1;;
  esac
  FORK=$S/results/${MASTER%-master}-$card-cupy
  [ -d $FORK ] || /scratch/qsh226/conda_envs/g4p/bin/python $S/lcc/fork_campaign.py $S/results/$MASTER $FORK
  PREV=""
  for i in $(seq 1 ${CHAIN:-1}); do
    DEP=""; [ -n "$PREV" ] && DEP="--dependency=afterany:$PREV"
    PREV=$(sbatch --parsable $DEP --export=ALL,CONTRACT_ENGINE=cupy,CUPY_ACCELERATORS=cub,CUPY_CACHE_DIR=$CACHE/cupy-nocut,QM_SUITE_DEVICE_ADMISSION=advisory,REPS=$REPS -p $PART -J ${MASTER%-master}-$card-cupy $S/lcc/gpu.sbatch $FORK all 100000 $CASES)
    echo "$card-cupy job $i: $PREV"
  done
done
