import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess(feats_df):
    """
    Recibe feats_df con columnas ['txId','time', feat1,feat2,...],
    y devuelve un tensor [N, F] con las features escaladas y tipo float32.
    """
    feat_cols = [c for c in feats_df.columns if c not in ("txId", "time")]
    print(f"Procesando {len(feat_cols)} columnas de características")

    X = feats_df[feat_cols].values.astype("float32")
    
    if np.isnan(X).any() or np.isinf(X).any():
        print("Advertencia: Valores NaN o Inf detectados. Reemplazando con ceros.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype("float32")
    
    return torch.from_numpy(Xs)