#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, unicodedata, csv, sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

# =====================================================
# SMAPE
# =====================================================
def smape_kaggle(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0

# =====================================================
# Text Normalization
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
# Numeric Feature Extraction
# =====================================================
def extract_numeric_feats(names):
    t = names.fillna("").astype(str).str.lower()

    def _to_float(x):
        try: return float(x)
        except: return 0
    
    gb = t.str.extract(r"(\d+(?:\.\d+)?)gb", expand=False).map(_to_float)
    inch = t.str.extract(r"(\d+(?:\.\d+)?)in", expand=False).map(_to_float)

    arr = np.column_stack([gb.fillna(0), inch.fillna(0)])
    arr = MaxAbsScaler().fit_transform(arr)
    return sparse.csr_matrix(arr)

# =====================================================
# Feature Builder
# =====================================================
def build_features(train_names, test_names):
    tr = normalize_name(train_names)
    te = normalize_name(test_names)

    tfw = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=3,
                          max_df=0.995, max_features=300000, sublinear_tf=True)
    tw_tr = tfw.fit_transform(tr)
    tw_te = tfw.transform(te)

    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,7), min_df=3,
                          max_df=0.995, max_features=300000, sublinear_tf=True)
    tc_tr = tfc.fit_transform(tr)
    tc_te = tfc.transform(te)

    num_tr = extract_numeric_feats(train_names)
    num_te = extract_numeric_feats(test_names)

    X_tr = sparse.hstack([tw_tr, tc_tr, num_tr]).tocsr()
    X_te = sparse.hstack([tw_te, tc_te, num_te]).tocsr()

    return X_tr, X_te

# =====================================================
# Main
# =====================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="train.csv")
    ap.add_argument("--test_path",  type=str, default="test.csv")
    ap.add_argument("--out_path",   type=str, default="submission.csv")
    args = ap.parse_args()

    tr = pd.read_csv(args.train_path)
    te = pd.read_csv(args.test_path)

    y = tr["price"].astype(float).values
    y_log = np.log1p(y)

    print(">> Building Features...")
    X_tr, X_te = build_features(tr["name"], te["name"])
    print("Features shape:", X_tr.shape, X_te.shape)

    print(">> Train Model A (Ridge, alpha=1.0)")
    modelA = Ridge(alpha=1.0)
    modelA.fit(X_tr, y_log)
    predA_tr = np.expm1(modelA.predict(X_tr))
    predA_te = np.expm1(modelA.predict(X_te))

    print(">> Train Model B (Ridge, alpha=3.0)")
    modelB = Ridge(alpha=3.0)
    modelB.fit(X_tr, y_log)
    predB_tr = np.expm1(modelB.predict(X_tr))
    predB_te = np.expm1(modelB.predict(X_te))

    X_blend_tr = np.column_stack([predA_tr, predB_tr])
    X_blend_te = np.column_stack([predA_te, predB_te])

    print(">> Train LightGBM Blender")
    blender = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    blender.fit(X_blend_tr, y)

    final_pred = blender.predict(X_blend_te)
    final_pred = np.clip(final_pred, 0, None)
    
    # === 評估模型（使用訓練集的預測結果）===
    final_pred_train = blender.predict(X_blend_tr)
    final_pred_train = np.clip(final_pred_train, 0, None)

    SMAPE = smape_kaggle(y, final_pred_train)
    MAE = mean_absolute_error(y, final_pred_train)

    # 舊版本 sklearn：手動 RMSE
    MSE = mean_squared_error(y, final_pred_train)
    RMSE = MSE ** 0.5

    print("\n=== 訓練集預估表現（內估）===")
    print(f"SMAPE: {SMAPE:.2f}%")
    print(f"MAE  : {MAE:.2f}")
    print(f"RMSE : {RMSE:.2f}")

    sub = pd.DataFrame({"name": te["name"], "price": final_pred})
    sub.to_csv(args.out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print("✅ Saved:", args.out_path)


if __name__ == "__main__":
    main()