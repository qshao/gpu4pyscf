#!/bin/bash
# usage: launch_capacity.sh CARD...   (chain of CHAIN jobs per card, default 10)
S=/scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
CACHE=/scratch/qsh226/g4p_lcc/cache
for card in "$@"; do
  case $card in
    a100)    PART=A2V80_ICE56M256_L;  ENG="ALL";;
    v100)    PART=V4V32_SKY32M192_L;  ENG="ALL";;
    v100_16) PART=V4V16_SKY32M192_L;  ENG="ALL";;
    p100_16) PART=P4V16_HAS16M128_L;  ENG="ALL,CONTRACT_ENGINE=cupy,CUPY_ACCELERATORS=cub,CUPY_CACHE_DIR=$CACHE/cupy-sm60";;
    p100_12) PART=P4V12_SKY32M192_L;  ENG="ALL,CONTRACT_ENGINE=cupy,CUPY_ACCELERATORS=cub,CUPY_CACHE_DIR=$CACHE/cupy-sm60";;
  esac
  PREV=""
  for i in $(seq 1 ${CHAIN:-10}); do
    DEP=""; [ -n "$PREV" ] && DEP="--dependency=afterany:$PREV"
    PREV=$(sbatch --parsable $DEP --export=$ENG,QM_SUITE_DEVICE_ADMISSION=advisory -p $PART -J cap-$card $S/lcc/capacity.sbatch $card)
    [ $i = 1 ] && echo "$card first job: $PREV"
  done
done
