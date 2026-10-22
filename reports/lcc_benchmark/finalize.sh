#!/bin/bash
# Wait for the 24-thread CPU lane, audit all strata with both CPU lanes, rebuild data, figures, tables and the PDF.
set -e
export PATH=/opt/ohpc/pub/libs/ccs/texlive/2020/bin/x86_64-linux:$PATH
PY=/scratch/qsh226/conda_envs/g4p/bin/python
R=/scratch/qsh226/gpu4pyscf/reports/lcc_benchmark
S=/scratch/qsh226/gpu4pyscf/benchmarks/single_gpu/qm_suite
until [ -z "$(squeue -h -j 6272341 2>/dev/null)" ]; do sleep 60; done
JOB=$(cd $S && sbatch --parsable lcc/audit_report.sbatch /scratch/qsh226/g4p_lcc/campaign1_final a100=campaign1-lcc-a100 v100=campaign1-lcc-v100 p100_16=campaign1-lcc-p100_16 p100_12=campaign1-lcc-p100_12 a100_cupy=campaign1-lcc-a100-cupy v100_cupy=campaign1-lcc-v100-cupy cpu=cpu48=campaign1-lcc-cpu cpu=cpu24b1=campaign1-lcc-cpu-24b1)
until [ -z "$(squeue -h -j $JOB 2>/dev/null)" ]; do sleep 30; done
cp /scratch/qsh226/g4p_lcc/campaign1_final.json $R/data/campaign.json
cd $R
$PY collect_aux.py | tail -1
$PY add_sizes.py 2>&1 | tail -1
$PY make_tables.py | tail -1
$PY make_figures.py 2>&1 | grep -E "FAILED|Error" || true
for i in 1 2 3 4; do pdflatex -interaction=nonstopmode report.tex > build.log 2>&1 || true; done
cp report.pdf GPU4pyscf_LCC_benchmark_report.pdf
grep -E "Overfull|undefined|^!" build.log | head -5 || true
echo "FINALIZED $(date)"
