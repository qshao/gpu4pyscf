#!/bin/bash
# usage: launch_cards.sh MASTER_NAME CASE_LIST_FILE REPS [CARDS...]
# Forks MASTER per card (identical campaign.json, shared CPU references) and submits the scheduler for each.
# Cards: a100 v100 p100_16 p100_12 (default: all four). P100 uses the CuPy-fallback engine.
set -e
MASTER=$1; CASES=$2; REPS=$3; shift 3
CARDS=${@:-a100 v100 p100_16 p100_12}
S=/scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
CACHE=/scratch/qsh226/g4p_lcc/cache
for card in $CARDS; do
  case $card in
    a100)    PART=A2V80_ICE56M256_L;     ENG="ALL";;
    v100)    PART=V4V32_SKY32M192_L;     ENG="ALL";;
    p100_16) PART=P4V16_HAS16M128_L;     ENG="ALL,CONTRACT_ENGINE=cupy,CUPY_ACCELERATORS=cub,CUPY_CACHE_DIR=$CACHE/cupy-sm60";;
    p100_12) PART=P4V12_SKY32M192_L;     ENG="ALL,CONTRACT_ENGINE=cupy,CUPY_ACCELERATORS=cub,CUPY_CACHE_DIR=$CACHE/cupy-sm60";;
    *) echo "unknown card $card"; exit 1;;
  esac
  FORK=$S/results/${MASTER%-master}-$card
  [ -d $FORK ] || /scratch/qsh226/conda_envs/g4p/bin/python $S/lcc/fork_campaign.py $S/results/$MASTER $FORK
  # Chain of CHAIN jobs: a follow-up starts after the previous one ends (3-day limit); once everything is done a follow-up exits at once.
  PREV=""
  for i in $(seq 1 ${CHAIN:-1}); do
    DEP=""; [ -n "$PREV" ] && DEP="--dependency=afterany:$PREV"
    PREV=$(sbatch --parsable $DEP --export=$ENG,QM_SUITE_DEVICE_ADMISSION=advisory,REPS=$REPS -p $PART -J ${MASTER%-master}-$card $S/lcc/gpu.sbatch $FORK all 100000 $CASES)
    echo "$card job $i: $PREV"
  done
done
