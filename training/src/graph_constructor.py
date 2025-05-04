import torch
from torch_geometric.data import Data
def build_graph(X, edges_df, labels_df):
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
    y = torch.from_numpy(labels_df["class"].fillna(0).to_numpy()).long()
    return Data(x=X, edge_index=edge_index, y=y)
