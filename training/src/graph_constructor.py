import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from .config import NUM_FEATURES

# Mapeo coherente: 1 = ilícito (fraude), 2/unknown = lícito
LABEL_MAP = {"1": 1, "2": 0, "unknown": 0}

def build_graph(feats_df, X, edges_df, labels_df) -> Data:
    tx_ids = feats_df["txId"].astype(str).tolist()
    id2idx = {tx: idx for idx, tx in enumerate(tx_ids)}

    # --- aristas -----------------------------------------------------------
    s_col, d_col = edges_df.columns[:2]
    src = edges_df[s_col].astype(str)
    dst = edges_df[d_col].astype(str)

    valid = [(s, d) for s, d in zip(src, dst) if s in id2idx and d in id2idx]
    if not valid:
        raise ValueError("No se encontraron aristas válidas.")

    src_idx = [id2idx[s] for s, _ in valid]
    dst_idx = [id2idx[d] for _, d in valid]
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    # --- etiquetas ---------------------------------------------------------
    N = len(tx_ids)
    y = torch.zeros(N, dtype=torch.float32)

    for _, row in labels_df.iterrows():
        tx = str(row["txId"])
        cls = str(row["class"])
        if tx in id2idx and cls in LABEL_MAP:
            y[id2idx[tx]] = LABEL_MAP[cls]

    print(f"Etiquetas: ilícitas={int(y.sum())} / totales={N}")

    x = X if isinstance(X, torch.Tensor) else torch.from_numpy(X.astype(np.float32))
    assert x.shape[1] == NUM_FEATURES, "Dimensión de features inconsistente"

    return Data(x=x, edge_index=edge_index, y=y)
