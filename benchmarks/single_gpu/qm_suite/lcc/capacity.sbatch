#!/bin/bash
#SBATCH -A gol_qsh226_uksr
#SBATCH -N 1 -n 1 -c 16 --gres=gpu:1 --mem=0 -t 3-00:00:00
#SBATCH -o /scratch/qsh226/g4p_lcc/logs/%x-%j.log
# Capacity ladder for one card (resumable; exit code 3 = out of walltime, the next job in the chain continues).
# usage: sbatch -p PART -J NAME capacity.sbatch CARD
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
export OMP_NUM_THREADS=4
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc
python capacity.py $1
