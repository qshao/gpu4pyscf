#!/bin/bash
#SBATCH -J qm-audit
#SBATCH -A col_qsh226_uksr
#SBATCH -p CAC48M192_L
#SBATCH -N 1 -c 8 --mem=64G -t 12:00:00
#SBATCH -o /scratch/qsh226/g4p_lcc/logs/%x-%j.log
# usage: sbatch audit_report.sbatch OUT_PREFIX card=fork [card=fork ...] cpu=fork
source /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc/env.sh
cd /scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite/lcc
OUT=$1; shift
ARGS=""
for a in "$@"; do case $a in cpu=*) ARGS="$ARGS --cpu ${a#cpu=}";; *) ARGS="$ARGS --card $a";; esac; done
python compare_variants.py campaign1 $OUT $ARGS
