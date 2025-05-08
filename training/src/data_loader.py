import os
import pandas as pd
from .config import DATA_PATH, NUM_FEATURES

def load_raw():
    """
    Devuelve tres DataFrames:
      - feats_df  : ['txId','time', feat1 … feat166]
      - edges_df  : ['txId1','txId2']
      - labels_df : ['txId','class']
    """
    feats_path = os.path.join(DATA_PATH, "elliptic_txs_features.csv")

    # -------- calcular nº de columnas -------------
    with open(feats_path, "r") as f:
        feature_count = len(f.readline().strip().split(","))   # 168 total
    assert feature_count - 2 == NUM_FEATURES, \
        f"Se esperaban {NUM_FEATURES} features, pero el CSV tiene {feature_count-2}"

    feature_columns = ["txId", "time"] + \
                      [f"feat{i}" for i in range(1, NUM_FEATURES + 1)]

    feats_df  = pd.read_csv(feats_path, header=None, names=feature_columns)
    edges_df  = pd.read_csv(os.path.join(DATA_PATH, "elliptic_txs_edgelist.csv"))
    labels_df = pd.read_csv(os.path.join(DATA_PATH, "elliptic_txs_classes.csv"))

    print(f"Features : {feats_df.shape}")
    print(f"Edges    : {edges_df.shape}")
    print(f"Labels   : {labels_df.shape}")

    return feats_df, edges_df, labels_df
