#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, csv
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
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
    tw_tr = tfw.fit_transform(tr); tw_te = tfw.transform(te)

    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,7), min_df=3,
                          max_df=0.995, max_features=300000, sublinear_tf=True)
    tc_tr = tfc.fit_transform(tr); tc_te = tfc.transform(te)

    num_tr = extract_numeric_feats(train_names)
    num_te = extract_numeric_feats(test_names)

    X_tr = sparse.hstack([tw_tr, tc_tr, num_tr]).tocsr()
    X_te = sparse.hstack([tw_te, tc_te, num_te]).tocsr()
    return X_tr, X_te

# =====================================================
# Evaluate One Combination
# =====================================================
def evaluate_combo(combo_name, ridge_alphas, lgb_params, X_train, X_valid, y_train, y_valid):
    print(f"\n=== {combo_name} ===")

    preds_train = []
    preds_valid = []

    # 第一層 Ridge 模型（可多個 alpha）
    for alpha in ridge_alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, np.log1p(y_train))
        preds_train.append(np.expm1(model.predict(X_train)))
        preds_valid.append(np.expm1(model.predict(X_valid)))

    X_blend_train = np.column_stack(preds_train)
    X_blend_valid = np.column_stack(preds_valid)

    # 第二層 LGBM blender
    blender = lgb.LGBMRegressor(random_state=42, **lgb_params)
    blender.fit(X_blend_train, y_train)

    pred_valid = np.clip(blender.predict(X_blend_valid), 0, None)

    smape = smape_kaggle(y_valid, pred_valid)
    mae = mean_absolute_error(y_valid, pred_valid)
    rmse = mean_squared_error(y_valid, pred_valid)**0.5

    print(f"SMAPE: {smape:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    return smape, mae, rmse

# =====================================================
# Main
# =====================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="train.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.train_path)
    X, _ = build_features(df["name"], df["name"])
    y = df["price"].astype(float).values

    # 分出訓練/驗證集
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    # 組合 1：最終
    results.append(("Combo 1 (Final)", *evaluate_combo(
        "Combo 1 (Final)",
        ridge_alphas=[1.0, 3.0],
        lgb_params=dict(n_estimators=600, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9),
        X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid
    )))

    # 組合 2：簡化 L2
    results.append(("Combo 2 (Simplified L2)", *evaluate_combo(
        "Combo 2 (Simplified L2)",
        ridge_alphas=[1.0, 3.0],
        lgb_params=dict(n_estimators=200, learning_rate=0.10, subsample=0.9, colsample_bytree=0.9),
        X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid
    )))

    # 組合 3：單一 L1
    results.append(("Combo 3 (Single L1)", *evaluate_combo(
        "Combo 3 (Single L1)",
        ridge_alphas=[2.0],
        lgb_params=dict(n_estimators=600, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9),
        X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid
    )))

    print("\n=== Summary (Validation) ===")
    print("{:<28s} {:>8s} {:>8s} {:>8s}".format("Combo", "SMAPE", "MAE", "RMSE"))
    for name, smape, mae, rmse in results:
        print("{:<28s} {:>8.2f} {:>8.2f} {:>8.2f}".format(name, smape, mae, rmse))

if __name__ == "__main__":
    main()