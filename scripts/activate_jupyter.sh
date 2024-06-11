#!/bin/bash

# 引数のチェック
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [cpu|gpu]"
    exit 1
fi

# 変数の定義
SIF_PATH="/data10/imageshare/kinuki/cuda116_py39_jupyter.sif"
VENV_PATH="/data10/kinuki/da_rankine/.venv"
CMD="source $VENV_PATH/bin/activate && poetry run jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root"

# CPUモードとGPUモードの設定
if [ "$1" == "cpu" ]; then
    srun --chdir $(pwd) singularity exec $SIF_PATH /bin/bash -c "$CMD"
elif [ "$1" == "gpu" ]; then
    # srun --chdir $(pwd) --gpus 20gb:1 -p gpu singularity exec --nv $SIF_PATH /bin/bash -c "$CMD"
    srun --chdir $(pwd) --gpus 40gb:1 -p gpu singularity exec --nv $SIF_PATH /bin/bash -c "$CMD"
else
    echo "Invalid argument: $1. Use 'cpu' or 'gpu'."
    exit 1
fi