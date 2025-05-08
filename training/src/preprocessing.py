import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, pathlib, os
from .config import NUM_FEATURES

def preprocess(feats_df, *, save_scaler=False):
    """
    Recibe feats_df y devuelve tensor [N, F] float32.
    Si save_scaler=True serializa el StandardScaler en model/scaler.pkl
    """
    feat_cols = [c for c in feats_df.columns if c not in ("txId", "time")]
    assert len(feat_cols) == NUM_FEATURES, \
        f"Se esperaban {NUM_FEATURES} features y hay {len(feat_cols)}."

    print(f"Procesando {len(feat_cols)} columnas de características")

    X = feats_df[feat_cols].values.astype("float32")

    # Limpieza de NaN/Inf
    if np.isnan(X).any() or np.isinf(X).any():
        print("Advertencia: Valores NaN o Inf detectados. Reemplazando con ceros.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler().fit(X)
    Xs     = scaler.transform(X).astype("float32")

    if save_scaler:
        pathlib.Path("model").mkdir(exist_ok=True)
        joblib.dump(scaler, os.path.join("model", "scaler.pkl"))
        print("Scaler guardado en model/scaler.pkl")

    return torch.from_numpy(Xs)
