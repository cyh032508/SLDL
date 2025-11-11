# Miniproject1

---

```bash
# 資料夾結構
miniproject1/
├── inference.sh
├── models
│   ├── blender.txt
│   ├── modelA.joblib
│   ├── modelB.joblib
│   ├── num_scaler.joblib
│   ├── tfc.joblib
│   └── tfw.joblib
├── README.md
├── requirements.txt
├── venv/
└── src
    ├── inference.py
    └── train.py
```

## 1.  套件安裝 （應該都已經都用好，可跳過）

```bash
cd miniproject1/

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

## 2. 模型

訓練好的模型已經存放在 `models/` 中，如果要重新訓練模型可使用以下指令

```bash
cd ~/miniproject1
python src/train.py --train_path train.csv --model_dir models
# train.csv 請自行指定路徑
```

訓練好的模型應該會輸出在 `models/` 中。

---

## 3.  使用 (Usage / Inference)

```bash

```
