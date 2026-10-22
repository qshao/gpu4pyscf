#!/bin/bash
#SBATCH -J qm-prepare
#SBATCH -A col_qsh226_uksr
#SBATCH -p CAC48M192_L
#SBATCH -N 1 -c 8 --mem=64G -t 04:00:00
#SBATCH -o /scratch/qsh226/g4p_lcc/logs/%x-%j.log
# usage: sbatch prepare.sbatch NAME CASE_LIST_FILE REPS
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
export QM_SUITE_REPETITIONS=$3 OMP_NUM_THREADS=8
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
python lcc/prepare_campaign.py $1 $(cat $2 | tr '\n' ' ')
