import torch, torch.nn.functional as F, time, pathlib
from .data_loader       import load_raw
from .preprocessing     import preprocess
from .graph_constructor import build_graph
from .model_def         import FraudGraphSAGE
from .config            import HIDDEN, EPOCHS, NUM_FEATURES

def main():
    feats_df, edges_df, labels_df = load_raw()

    X = preprocess(feats_df, save_scaler=True)

    data  = build_graph(feats_df, X, edges_df, labels_df)
    model = FraudGraphSAGE(NUM_FEATURES, HIDDEN)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()

        out = model(data.x, data.edge_index)        # [N] probabilidades
        loss = F.binary_cross_entropy(out, data.y)

        loss.backward()
        opt.step()

        preds = (out > 0.5).float()
        acc   = (preds == data.y).float().mean()

        print(f"Época {epoch+1}/{EPOCHS}  Pérdida={loss:.4f}  Precisión={acc:.4f}")

    pathlib.Path("model").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "model/fraud_gnn.pt")
    print("Modelo guardado en model/fraud_gnn.pt")

if __name__ == "__main__":
    tic = time.time()
    main()
    print("⏱", round(time.time() - tic, 1), "s")
