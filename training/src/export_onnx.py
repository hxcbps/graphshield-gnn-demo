import torch, pathlib, os
from .model_def         import FraudGraphSAGE
from .config            import HIDDEN, NUM_FEATURES
from .data_loader       import load_raw
from .preprocessing     import preprocess
from .graph_constructor import build_graph

def export_dynamic():
    feats_df, edges_df, labels_df = load_raw()
    X    = preprocess(feats_df)                
    data = build_graph(feats_df, X, edges_df, labels_df)

    model = FraudGraphSAGE(NUM_FEATURES, HIDDEN)
    model.load_state_dict(torch.load("model/fraud_gnn.pt"))
    model.eval()

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    pathlib.Path(model_dir).mkdir(exist_ok=True)
    onnx_path = os.path.join(model_dir, "fraud_gnn.onnx")

    print(f"Exportando modelo dinámico → {onnx_path}")

    dynamic_axes = {
        "x":          {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
        "prob":       {0: "num_nodes"},
    }

    torch.onnx.export(
        model,
        (data.x, data.edge_index),
        onnx_path,
        input_names=["x", "edge_index"],
        output_names=["prob"],
        dynamic_axes=dynamic_axes,
        opset_version=17
    )

    print("✅ Exportación completada")

if __name__ == "__main__":
    export_dynamic()
