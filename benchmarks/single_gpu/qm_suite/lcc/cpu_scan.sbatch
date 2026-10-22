#!/bin/bash
#SBATCH -J qm-cpuscan
#SBATCH -A col_qsh226_uksr
#SBATCH -p CAC48M192_L
#SBATCH -N 1 -n 1 --exclusive --mem=0 -t 1-12:00:00
#SBATCH -o /scratch/qsh226/g4p_lcc/logs/%x-%A_%a.log
# CPU thread-scaling experiment. task line: "CASE INDEX OMP_THREADS BLAS_THREADS BIND(none|close) NAME"
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
read CASE IDX OMP BLAS BIND NAME < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $2)
export OMP_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$BLAS MKL_NUM_THREADS=$BLAS CUDA_VISIBLE_DEVICES=
if [ "$BIND" = close ]; then export OMP_PROC_BIND=close OMP_PLACES=cores; fi
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
echo "node $(hostname) cpus $(nproc) case $CASE config $NAME omp=$OMP blas=$BLAS bind=$BIND"
python run.py --campaign-root $1 --case $CASE --variant CPU --mode timing --attempt $IDX --pair-index $IDX
