#!/bin/bash
#SBATCH -J qm-ref
#SBATCH -A col_qsh226_uksr
#SBATCH -p CAC48M192_L
#SBATCH -N 1 -n 1 -c 24 --mem=150G -t 2-00:00:00
#SBATCH -o /scratch/qsh226/g4p_lcc/logs/%x-%A_%a.log
# usage: sbatch --array=0-N reference.sbatch MASTER_CAMPAIGN_DIR CASE_LIST_FILE
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
MASTER=$1; CASE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $2)
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24   # CPU oracle thread count is recorded in the job
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
python run.py --campaign-root $MASTER --case $CASE --variant CPU --mode reference --attempt 0
