import torch
from torch_geometric.nn import SAGEConv
class FraudGraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden, classes=2):
        super().__init__()
        self.g1 = SAGEConv(in_dim, hidden)
        self.g2 = SAGEConv(hidden, hidden)
        self.out = torch.nn.Linear(hidden, classes)
    def forward(self, x, edge_index):
        x = self.g1(x, edge_index).relu()
        x = self.g2(x, edge_index).relu()
        return torch.softmax(self.out(x), dim=-1)[:, 1]  # prob fraude
