#!/bin/bash
#SBATCH -J qm-cpu
#SBATCH -A col_qsh226_uksr
#SBATCH -p CAC48M192_L
#SBATCH -N 1 -n 1 --exclusive --mem=0 -t 3-00:00:00
#SBATCH -o /scratch/qsh226/g4p_lcc/logs/%x-%A_%a.log
# CPU timing lane: one whole 48-core node, 48 threads, fresh process per run.
# usage: sbatch --array=0-N%K cpu_timing.sbatch CPU_FORK_DIR TASK_LIST   (task line: "CASE_ID REPETITION")
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
read CASE REP < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $2)
export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
echo "node $(hostname) cpus $(nproc) case $CASE repetition $REP"
python run.py --campaign-root $1 --case $CASE --variant CPU --mode timing --attempt $REP --pair-index $REP
