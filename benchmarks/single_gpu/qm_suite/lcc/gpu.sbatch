#!/bin/bash
#SBATCH -A gol_qsh226_uksr
#SBATCH -N 1 -n 1 -c 8 --gres=gpu:1 --mem=100G -t 3-00:00:00
#SBATCH -o /scratch/qsh226/g4p_lcc/logs/%x-%j.log
# usage: sbatch -p PARTITION -J NAME gpu.sbatch FORK_CAMPAIGN_DIR CAMPAIGN GPU_HOURS CASE_LIST_FILE
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python schedule.py --campaign $2 --output $1 --cases $(cat $4) --repetitions ${REPS:-5} --gpu-hours ${3:-8} --cpu-hours 100000 --timeout 7200
