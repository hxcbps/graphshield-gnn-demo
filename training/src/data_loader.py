# training/src/data_loader.py
import os
import pandas as pd
from .config import DATA_PATH

def load_raw():
    """
    Devuelve tres DataFrames con las cabeceras correctas:
      - feats_df: columnas ['txId','time', feat1,feat2,...]
      - edges_df: columnas ['txId1','txId2']
      - labels_df: columnas ['txId','class']
    """
    feats_path = os.path.join(DATA_PATH, "elliptic_txs_features.csv")

    with open(feats_path, 'r') as f:
        first_line = f.readline().strip()
        feature_count = len(first_line.split(','))
    
    feature_columns = ['txId', 'time'] + [f'feat{i}' for i in range(1, feature_count-1)]
    
    feats_df = pd.read_csv(feats_path, header=None, names=feature_columns)

    edges_df = pd.read_csv(os.path.join(DATA_PATH, "elliptic_txs_edgelist.csv"))
    labels_df = pd.read_csv(os.path.join(DATA_PATH, "elliptic_txs_classes.csv"))
    
    print(f"Features: {feats_df.shape[0]} filas, {feats_df.shape[1]} columnas")
    print(f"Edges: {edges_df.shape[0]} filas, {edges_df.shape[1]} columnas")
    print(f"Labels: {labels_df.shape[0]} filas, {labels_df.shape[1]} columnas")
    
    return feats_df, edges_df, labels_df