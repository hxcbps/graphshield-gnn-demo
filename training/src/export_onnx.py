# training/src/export_onnx.py

import torch
import pathlib
import os

from .model_def         import FraudGraphSAGE
from .config            import HIDDEN
from .data_loader       import load_raw
from .preprocessing     import preprocess
from .graph_constructor import build_graph

def export():

    feats_df, edges_df, labels_df = load_raw()
    X    = preprocess(feats_df)
    data = build_graph(feats_df, X, edges_df, labels_df)
    
    model = FraudGraphSAGE(X.size(1), HIDDEN)
    model.load_state_dict(torch.load("model/fraud_gnn.pt"))
    model.eval()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  
    model_dir = os.path.join(project_root, "model")
    
    pathlib.Path(model_dir).mkdir(exist_ok=True)

    onnx_path = os.path.join(model_dir, "fraud_gnn.onnx")
    
    print(f"Exportando modelo a: {onnx_path}")
    
    torch.onnx.export(
        model,
        (data.x, data.edge_index),
        onnx_path,
        input_names = ["x","edge_index"],
        output_names= ["prob"],
        opset_version=17
    )
    
    print(f"✅ Modelo exportado correctamente a: {onnx_path}")

if __name__ == "__main__":
    export()