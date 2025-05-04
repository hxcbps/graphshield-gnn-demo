import os, pandas as pd
from .config import DATA_PATH
def load_raw():
    feats  = pd.read_csv(os.path.join(DATA_PATH, "elliptic_txs_features.csv"), header=None)
    edges  = pd.read_csv(os.path.join(DATA_PATH, "elliptic_txs_edgelist.csv"))
    labels = pd.read_csv(os.path.join(DATA_PATH, "elliptic_txs_classes.csv"))
    return feats, edges, labels
