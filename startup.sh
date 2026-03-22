#!/bin/bash
source ../proj-env/bin/activate

export CUDA_HOME=$TACC_CUDA_DIR
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Let the active Python report its own header location.
# This overrides the hardcoded /usr/include/python3.9 that nvcc was picking up.
PY_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
export CPLUS_INCLUDE_PATH=$PY_INCLUDE:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$PY_INCLUDE:$C_INCLUDE_PATH

USE_NINJA=0 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="9.0" python setup.py build_ext --inplace
echo "---Done---"