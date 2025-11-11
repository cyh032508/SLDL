#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, unicodedata, csv, sys, os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib # 用來儲存 sklearn 元件

# =====================================================
# SMAPE (用來驗證)
# =====================================================
def smape_kaggle(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0

# =====================================================
# Text Normalization (和 inference.py 保持一致)
# =====================================================
def normalize_digits_smart(x):
    protected = {}
    def hold(m):
        key = f"@@U{len(protected)}@@"
        protected[key] = m.group(0)
        return key
    x2 = re.sub(r"\b\d+(?:\.\d+)?(gb|tb|mb|in|l|kg|g|w|mah)\b", hold, x, flags=re.I)
    x2 = re.sub(r"\b\d{3,5}[x×]\d{3,5}\b", hold, x2)
    x2 = re.sub(r"\b\d{1,2}:\d{1,2}\b", hold, x2)
    x2 = re.sub(r"\b\d+\s*(pcs|入|x)\b", hold, x2, flags=re.I)
    def repl(m):
        s = m.group(0)
        if len(s)>=4: return "<D4>"
        if len(s)==3: return "<D3>"
        if len(s)==2: return "<D2>"
        return s
    x2 = re.sub(r"\b\d{2,}\b", repl, x2)
    for k,v in protected.items(): x2 = x2.replace(k, v)
    return x2

def normalize_name(s):
    s = s.fillna("").astype(str)
    s = s.str.lower()
    s = s.str.replace(r"英寸|吋", "in", regex=True)
    s = s.str.replace(r"公斤","kg").str.replace("公克","g")
    s = s.apply(normalize_digits_smart)
    s = s.str.replace(r"\s+"," ", regex=True).str.strip()
    return s

# =====================================================
# Numeric Feature Extraction (和 inference.py 保持一致)
# =====================================================
def extract_numeric_feats(names, scaler=None):
    t = names.fillna("").astype(str).str.lower()

    def _to_float(x):
        try: return float(x)
        except: return 0
    
    gb = t.str.extract(r"(\d+(?:\.\d+)?)gb", expand=False).map(_to_float)
    inch = t.str.extract(r"(\d+(?:\.\d+)?)in", expand=False).map(_to_float)

    arr = np.column_stack([gb.fillna(0), inch.fillna(0)])
    
    if scaler is None:
        scaler = MaxAbsScaler()
        arr = scaler.fit_transform(arr)
    else:
        arr = scaler.transform(arr)
        
    return sparse.csr_matrix(arr), scaler

# =====================================================
# Main
# =====================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="../../train.csv")
    ap.add_argument("--model_dir",  type=str, default="../models")
    args = ap.parse_args()

    # --- 建立模型儲存資料夾 ---
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"Models will be saved to: {args.model_dir}")

    tr = pd.read_csv(args.train_path)
    y = tr["price"].astype(float).values
    y_log = np.log1p(y)

    print(">> Building Features & Saving Vectorizers...")
    
    # 1. Normalize
    tr_names_raw = tr["name"] # 保留原始 name 給 extract_numeric_feats
    tr_names_norm = normalize_name(tr_names_raw)

    # 2. TF-IDF Word
    tfw = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=3,
                          max_df=0.995, max_features=300000, sublinear_tf=True)
    tw_tr = tfw.fit_transform(tr_names_norm)
    joblib.dump(tfw, os.path.join(args.model_dir, "tfw.joblib"))
    print("Saved tfw.joblib")

    # 3. TF-IDF Char
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,7), min_df=3,
                          max_df=0.995, max_features=300000, sublinear_tf=True)
    tc_tr = tfc.fit_transform(tr_names_norm)
    joblib.dump(tfc, os.path.join(args.model_dir, "tfc.joblib"))
    print("Saved tfc.joblib")

    # 4. Numeric Features
    num_tr, num_scaler = extract_numeric_feats(tr_names_raw, scaler=None) # 傳入 None 讓他 fit
    joblib.dump(num_scaler, os.path.join(args.model_dir, "num_scaler.joblib"))
    print("Saved num_scaler.joblib")

    # 5. Combine
    X_tr = sparse.hstack([tw_tr, tc_tr, num_tr]).tocsr()
    print("Features shape:", X_tr.shape)

    # --- 訓練 Model A ---
    print(">> Train Model A (Ridge, alpha=1.0)")
    modelA = Ridge(alpha=1.0)
    modelA.fit(X_tr, y_log)
    predA_tr = np.expm1(modelA.predict(X_tr))
    joblib.dump(modelA, os.path.join(args.model_dir, "modelA.joblib"))
    print("Saved modelA.joblib")

    # --- 訓練 Model B ---
    print(">> Train Model B (Ridge, alpha=3.0)")
    modelB = Ridge(alpha=3.0)
    modelB.fit(X_tr, y_log)
    predB_tr = np.expm1(modelB.predict(X_tr))
    joblib.dump(modelB, os.path.join(args.model_dir, "modelB.joblib"))
    print("Saved modelB.joblib")

    # --- 訓練 Blender ---
    X_blend_tr = np.column_stack([predA_tr, predB_tr])
    print(">> Train LightGBM Blender")
    blender = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    blender.fit(X_blend_tr, y)
    blender.booster_.save_model(os.path.join(args.model_dir, "blender.txt"))
    print("Saved blender.txt")

    # --- 驗證 (同你原始碼) ---
    final_pred_train = blender.predict(X_blend_tr)
    final_pred_train = np.clip(final_pred_train, 0, None)
    SMAPE = smape_kaggle(y, final_pred_train)
    MAE = mean_absolute_error(y, final_pred_train)
    RMSE = mean_squared_error(y, final_pred_train) ** 0.5
    print("\n=== 訓練集預估表現 (SMAPE) ===")
    print(f"SMAPE: {SMAPE:.2f}% | MAE: {MAE:.2f} | RMSE: {RMSE:.2f}")
    print("✅ Training complete. Models saved in 'models/' folder.")

if __name__ == "__main__":
    main()