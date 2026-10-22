# Shared environment for LCC jobs: nothing in $HOME, one CUDA 12.4 / sm70+80+90 build.
E=/scratch/qsh226/conda_envs/g4p
export PATH=$E/bin:$PATH
export LD_LIBRARY_PATH=$E/lib:$E/lib/python3.11/site-packages/cutensor/lib:$LD_LIBRARY_PATH
export CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-/scratch/qsh226/g4p_lcc/cache/cupy} TMPDIR=/scratch/qsh226/g4p_lcc/cache/tmp XDG_CACHE_HOME=/scratch/qsh226/g4p_lcc/cache
export CUDA_PATH=/scratch/qsh226/g4p_lcc/cuda_path CUPY_ACCELERATORS=${CUPY_ACCELERATORS:-cub,cutensor}
export QM_SUITE_SCRATCH=/scratch/qsh226/tmp_cderi
mkdir -p $CUPY_CACHE_DIR $TMPDIR $QM_SUITE_SCRATCH
