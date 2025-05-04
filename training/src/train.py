import torch, torch.nn.functional as F, time, pathlib
from .data_loader import load_raw
from .preprocessing import preprocess
from .graph_constructor import build_graph
from .model_def import FraudGraphSAGE
from .config import *

def main():
    feats, edges, labels = load_raw()
    X = preprocess(feats)
    data = build_graph(X, edges, labels)
    model = FraudGraphSAGE(X.size(1), HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy(out, data.y.float())
        loss.backward(); opt.step()
        print(f"epoch {epoch+1}/{EPOCHS}  loss={loss.item():.4f}")
    pathlib.Path("model").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "model/fraud_gnn.pt")

if __name__ == "__main__":
    tic=time.time(); main(); print("⏱", round(time.time()-tic,1), "s")
