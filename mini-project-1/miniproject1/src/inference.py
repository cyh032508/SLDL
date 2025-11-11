#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, unicodedata, csv, sys, os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MaxAbsScaler
import lightgbm as lgb
import joblib

# =====================================================
# Text Normalization (和 train.py 保持一致)
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
# Numeric Feature Extraction (和 train.py 保持一致)
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
        raise ValueError("Scaler must be pre-fitted for inference.")
    else:
        arr = scaler.transform(arr)
        
    return sparse.csr_matrix(arr), scaler

# =====================================================
# Main Inference
# =====================================================
def main():
    ap = argparse.ArgumentParser()
    # 接收來自 inference.sh 的參數
    ap.add_argument("--test_path",  type=str, required=True)
    ap.add_argument("--output_path",type=str, required=True)
    ap.add_argument("--model_dir",  type=str, required=True)
    args = ap.parse_args()

    print(f">> Loading test data from: {args.test_path}")
    te = pd.read_csv(args.test_path)

    print(f">> Loading models from: {args.model_dir}")
    # --- 載入所有儲存的元件 ---
    tfw = joblib.load(os.path.join(args.model_dir, "tfw.joblib"))
    tfc = joblib.load(os.path.join(args.model_dir, "tfc.joblib"))
    num_scaler = joblib.load(os.path.join(args.model_dir, "num_scaler.joblib"))
    modelA = joblib.load(os.path.join(args.model_dir, "modelA.joblib"))
    modelB = joblib.load(os.path.join(args.model_dir, "modelB.joblib"))
    
    # LightGBM 載入
    model_path = os.path.join(args.model_dir, "blender.txt")
    blender = lgb.Booster(model_file=model_path)

    print(">> Building features for test set...")
    # 1. Normalize
    te_names_raw = te["name"]
    te_names_norm = normalize_name(te_names_raw)

    # 2. Transform (使用載入的 vectorizer)
    tw_te = tfw.transform(te_names_norm)
    tc_te = tfc.transform(te_names_norm)
    
    # 3. Numeric (使用載入的 scaler)
    num_te, _ = extract_numeric_feats(te_names_raw, scaler=num_scaler)

    # 4. Combine
    X_te = sparse.hstack([tw_te, tc_te, num_te]).tocsr()
    print("Test features shape:", X_te.shape)

    print(">> Generating predictions...")
    # --- Model A 預測 ---
    predA_te = np.expm1(modelA.predict(X_te))

    # --- Model B 預測 ---
    predB_te = np.expm1(modelB.predict(X_te))

    # --- Blender 預測 ---
    X_blend_te = np.column_stack([predA_te, predB_te])
    final_pred = blender.predict(X_blend_te)
    final_pred = np.clip(final_pred, 0, None)

    # --- 儲存結果 ---
    sub = pd.DataFrame({"name": te["name"], "price": final_pred})
    sub.to_csv(args.output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✅ Saved predictions to: {args.output_path}")

if __name__ == "__main__":
    main()