#!/bin/bash
#SBATCH -A gol_qsh226_uksr
#SBATCH -N 1 -n 1 -c 8 --gres=gpu:1 --mem=64G -t 04:00:00
#SBATCH -o /scratch/qsh226/g4p_lcc/logs/%x-%j.log
# Cold-start lane: the first-ever process (empty kernel cache) and a warm repeat, per variant and engine, on three sizes.
# usage: sbatch -p PART -J NAME coldstart.sbatch FORK_DIR "ENGINES"      ENGINES: space-separated subset of "cutensor cupy"
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
export OMP_NUM_THREADS=4
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
CASES="core-h2o-b3lyp-svp core-vitamin_c-b3lyp-svp core-tamoxifen-b3lyp-svp"
index=0
for engine in $2; do
  for variant in G0 G1; do
    FRESH=$(mktemp -d /scratch/qsh226/g4p_lcc/cache/cold-XXXXXX)      # empty kernel cache => first-ever process
    for state in cold warm; do
      for case in $CASES; do
        if [ $engine = cupy ]; then ENV="CONTRACT_ENGINE=cupy CUPY_ACCELERATORS=cub"; else ENV="CUPY_ACCELERATORS=cub,cutensor"; fi
        # the cold run of the first case compiles the kernels; later cases and the warm pass reuse the cache, so the cold
        # lane is only the very first process per (variant, engine): cold = first case, first pass.
        echo "== $engine $variant $state $case (pair index $index)"
        env $ENV CUPY_CACHE_DIR=$FRESH python run.py --campaign-root $1 --case $case --variant $variant --mode timing --attempt $index --pair-index $index
        index=$((index+1))
      done
    done
    rm -rf $FRESH
  done
done
