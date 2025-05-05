import torch
import torch.nn.functional as F
import time
import pathlib

from .data_loader       import load_raw
from .preprocessing     import preprocess
from .graph_constructor import build_graph
from .model_def         import FraudGraphSAGE
from .config            import *
 
def main():
    feats_df, edges_df, labels_df = load_raw()

    X = preprocess(feats_df)
    
    data = build_graph(feats_df, X, edges_df, labels_df)
    
    model = FraudGraphSAGE(X.size(1), HIDDEN)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-3)
    
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        
        out = model(data.x, data.edge_index)
        
        target = data.y.float() 

        loss = F.binary_cross_entropy(out, target)
        
        loss.backward()
        opt.step()
        
        predictions = (out > 0.5).float()
        accuracy = (predictions == target).float().mean()
        
        print(f"Época {epoch+1}/{EPOCHS}  Pérdida={loss.item():.4f}  Precisión={accuracy.item():.4f}")
    
    pathlib.Path("model").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "model/fraud_gnn.pt")
 
if __name__ == "__main__":
    tic = time.time()
    main()
    print("⏱", round(time.time()-tic,1), "s")