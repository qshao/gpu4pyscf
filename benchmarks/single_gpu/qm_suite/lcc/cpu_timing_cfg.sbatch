#!/bin/bash
#SBATCH -J qm-cpucfg
#SBATCH -A col_qsh226_uksr
#SBATCH -p CAC48M192_L
#SBATCH -N 1 -n 1 --exclusive --mem=0 -t 3-00:00:00
#SBATCH -o /scratch/qsh226/g4p_lcc/logs/%x-%A_%a.log
# CPU timing lane at a chosen threading configuration (whole node, so nothing else runs beside it).
# usage: sbatch --array=0-N%K cpu_timing_cfg.sbatch FORK TASK_LIST OMP_THREADS BLAS_THREADS
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
read CASE REP < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $2)
export OMP_NUM_THREADS=$3 OPENBLAS_NUM_THREADS=$4 MKL_NUM_THREADS=$4 CUDA_VISIBLE_DEVICES=
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
echo "node $(hostname) case $CASE repetition $REP omp=$3 blas=$4"
python run.py --campaign-root $1 --case $CASE --variant CPU --mode timing --attempt $REP --pair-index $REP
