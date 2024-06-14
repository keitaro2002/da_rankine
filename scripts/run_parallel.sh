#!/bin/bash

# 変数の定義
SIF_PATH="/data10/imageshare/cuda/cuda116_py39.sif"
VENV_PATH="/data10/kinuki/da_rankine/.venv"
PYTHON_SCRIPT="src/models/LETKF.py"  # 実行するPythonスクリプトのパスを指定

# ログディレクトリの作成
LOG_DIR="logs"
mkdir -p $LOG_DIR

# p_rangeの開始と終了の範囲を指定
START=0
END=$((91 * 91))
INTERVAL=50

# p_rangeを50区切りで作成して実行
for (( i=$START; i<$END; i+=$INTERVAL ))
do
  P_RANGE_START=$i
  P_RANGE_END=$(($i+$INTERVAL-1))
  P_RANGE_END=$(($P_RANGE_END<$END?$P_RANGE_END:$END-1))

  LOG_FILE="$LOG_DIR/log_${P_RANGE_START}_${P_RANGE_END}.txt"

  srun --chdir $(pwd) -p cpu singularity run $SIF_PATH /bin/bash -c "source $VENV_PATH/bin/activate && poetry run python $PYTHON_SCRIPT $P_RANGE_START $P_RANGE_END" > $LOG_FILE 2>&1 &
done

# 全てのバックグラウンドジョブが終了するのを待つ
wait
