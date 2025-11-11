#!/bin/bash
# -----------------------------------------------
# 專案要求：TA 會在 miniproject1/ 底下執行
# ./inference.sh --test_path /service/res/test.csv
# -----------------------------------------------

set -e # 任何指令失敗就立刻停止

echo "--- [1/4] Activating Virtual Environment ---"
# 假設你的 venv 叫 "venv" 且放在 miniproject1/
# 如果名稱或路徑不同，請修改 "venv/bin/activate"
source "venv/bin/activate" 

echo "--- [2/4] Running Inference Script ---"
# $1, $2 會是 "--test_path", "/service/res/test.csv"
# 我們把 $1 $2 (或 $*) 傳給 python
#
# 專案規定 output 必須是 miniproject1/predictions.csv
# 我們在 python 腳本中指定 --output_path "predictions.csv"
python3 src/inference.py \
    --model_dir "models" \
    --output_path "predictions.csv" \
    "$@"

echo "--- [3/4] Deactivating Virtual Environment ---"
deactivate

echo "--- [4/4] Inference Done. 'predictions.csv' created. ---"