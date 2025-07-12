from __future__ import annotations
import os, random, math, warnings
from pathlib import Path
from typing import Dict, List
from rdkit import RDLogger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np, pandas as pd, tensorflow as tf, joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, InputLayer

DATA_PATH = Path("ic50_2d_desc_fps_clean.csv")
SEED, FOLDS = 42, 5
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", message=r".*experimental_relax_shapes is deprecated.*", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", message=r"X does not have valid feature names.*", category=UserWarning, module="sklearn")

cfg_rf  = dict(n_estimators=500, random_state=SEED, n_jobs=-1)
cfg_lgb = dict(n_estimators=2000, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=SEED, objective="regression", verbose=-1)
cfg_mlp = dict(hidden_layer_sizes=(256,128), activation="relu", solver="adam", alpha=1e-4, learning_rate_init=1e-3, max_iter=400, early_stopping=True, random_state=SEED)
CNN_EPOCHS, CNN_BATCH = 100, 32

def to_pic50(nm: float) -> float:
    return 9.0 - math.log10(nm)

def metric_pack(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return r2_score(y_true, y_pred), rmse, mean_absolute_error(y_true, y_pred)

def build_cnn(shape: tuple[int,int]):
    model = Sequential([
        InputLayer(shape=shape),
        Conv1D(64, 3, padding="same", activation="relu"),
        Conv1D(32, 3, padding="same", activation="relu"),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model

df = pd.read_csv(DATA_PATH)
df["IC50_nM"] = pd.to_numeric(df["IC50_nM"], errors="coerce")
df = df[df["IC50_nM"] > 0].dropna(subset=["IC50_nM"])

y = df["IC50_nM"].apply(to_pic50).values

X_tab = (
    df.drop(columns=["IC50_nM", "Smiles"], errors="ignore")
      .select_dtypes(include=[np.number])
)
X_tab.replace([np.inf, -np.inf], np.nan, inplace=True)
X_tab.drop(columns=X_tab.columns[X_tab.isna().any()], inplace=True)
X_tab = X_tab.clip(-np.finfo(np.float32).max, np.finfo(np.float32).max).values.astype(np.float32)

results: Dict[str, List[tuple[float,float,float]]] = {k: [] for k in ["RF", "LGBM", "MLP", "CNN"]}

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for tr_idx, te_idx in kf.split(X_tab):
    X_tr, X_te, y_tr, y_te = X_tab[tr_idx], X_tab[te_idx], y[tr_idx], y[te_idx]

    # Random Forest
    rf = RandomForestRegressor(**cfg_rf).fit(X_tr, y_tr)
    results["RF"].append(metric_pack(y_te, rf.predict(X_te)))

    # LightGBM
    lgbm = LGBMRegressor(**cfg_lgb).fit(X_tr, y_tr)
    results["LGBM"].append(metric_pack(y_te, lgbm.predict(X_te)))

    # MLP
    mlp = Pipeline([("s", StandardScaler()), ("m", MLPRegressor(**cfg_mlp))])
    mlp.fit(X_tr, y_tr)
    results["MLP"].append(metric_pack(y_te, mlp.predict(X_te)))

    # 1‑D CNN
    scaler = StandardScaler().fit(X_tr)
    Xt_s, Xe_s = scaler.transform(X_tr)[..., None], scaler.transform(X_te)[..., None]
    cnn = build_cnn((Xt_s.shape[1], 1))
    cnn.fit(
        Xt_s, y_tr,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH,
        validation_split=0.2,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0)],
    )
    results["CNN"].append(metric_pack(y_te, cnn.predict(Xe_s, verbose=0).flatten()))

# Результат
print("Model  R2_mean±std  RMSE_mean±std  MAE_mean±std")
for name, vals in results.items():
    r2s, rmses, maes = zip(*vals)
    print(f"{name:<5} {np.mean(r2s):.3f}±{np.std(r2s):.3f}  {np.mean(rmses):.3f}±{np.std(rmses):.3f}  {np.mean(maes):.3f}±{np.std(maes):.3f}")

# Сохранение моделей
OUT_DIR = Path("models"); OUT_DIR.mkdir(exist_ok=True)

# Random Forest
rf_final = RandomForestRegressor(**cfg_rf).fit(X_tab, y)
joblib.dump(rf_final, OUT_DIR / "rf.joblib")

# LightGBM
lgbm_final = LGBMRegressor(**cfg_lgb).fit(X_tab, y)
joblib.dump(lgbm_final, OUT_DIR / "lgbm.joblib")

# MLP
mlp_final = Pipeline([("s", StandardScaler()), ("m", MLPRegressor(**cfg_mlp))])
mlp_final.fit(X_tab, y)
joblib.dump(mlp_final, OUT_DIR / "mlp.joblib")

# CNN
scaler_all = StandardScaler().fit(X_tab)
X_all_scaled = scaler_all.transform(X_tab)[..., None]
cnn_full = build_cnn((X_all_scaled.shape[1], 1))
cnn_full.fit(
    X_all_scaled, y,
    epochs=CNN_EPOCHS,
    batch_size=CNN_BATCH,
    validation_split=0.2,
    verbose=0,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0)],
)
cnn_full.save(OUT_DIR / "cnn.keras")
joblib.dump(scaler_all, OUT_DIR / "cnn_scaler.joblib")

print("Models saved to", OUT_DIR)
