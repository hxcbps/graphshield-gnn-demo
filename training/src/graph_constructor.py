import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

def build_graph(feats_df: pd.DataFrame,
                X: torch.Tensor,
                edges_df: pd.DataFrame,
                labels_df: pd.DataFrame) -> Data:
    """
    feats_df: DataFrame con 'txId' + time + feats...
    X:        Tensor [N, F] resultado de preprocess(feats_df)
    edges_df: DataFrame con dos cols [txId_src, txId_dst]
    labels_df: DataFrame con cols ['txId','class']
    
    Nota: Para binary_cross_entropy, las etiquetas deben estar entre 0 y 1.
    Mapeamos 'unknown' -> 0 y '2' -> 1
    """
    tx_ids = feats_df["txId"].astype(str).tolist()
    id2idx = {tx: idx for idx, tx in enumerate(tx_ids)}
    
    src_col, dst_col = edges_df.columns[0], edges_df.columns[1]
    src_list = edges_df[src_col].astype(str).tolist()
    dst_list = edges_df[dst_col].astype(str).tolist()
    
    valid_edges = [(s, d) for s, d in zip(src_list, dst_list) if s in id2idx and d in id2idx]
    if not valid_edges:
        raise ValueError("No se encontraron aristas válidas. Verifica que los IDs de transacción coincidan entre los archivos.")
    
    src_idx = [id2idx[s] for s, _ in valid_edges]
    dst_idx = [id2idx[d] for _, d in valid_edges]
    
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    
    N = len(tx_ids)
    y = torch.zeros(N, dtype=torch.long)
    
    unknown_count = 0
    legit_count = 0
    
    for _, row in labels_df.iterrows():
        tx = str(row["txId"])
        cls = row["class"]
        
        if pd.isna(cls) or tx not in id2idx:
            continue
            
        if cls == 'unknown':
            y[id2idx[tx]] = 0
            unknown_count += 1
        else:
            y[id2idx[tx]] = 1  
            legit_count += 1
    
    print(f"Etiquetas mapeadas: {unknown_count} 'unknown' -> 0, {legit_count} legítimas -> 1")
    
    x = X.float() if isinstance(X, torch.Tensor) else torch.from_numpy(X.astype(np.float32))
    
    return Data(x=x, edge_index=edge_index, y=y)