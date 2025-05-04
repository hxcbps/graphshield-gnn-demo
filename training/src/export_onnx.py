import torch, pathlib
from .model_def import FraudGraphSAGE
from .config import ONNX_OUT, HIDDEN
from .data_loader import load_raw
from .preprocessing import preprocess
from .graph_constructor import build_graph

def export():
    feats, edges, labels = load_raw()
    X = preprocess(feats)
    g = build_graph(X, edges, labels)
    model = FraudGraphSAGE(X.size(1), HIDDEN)
    model.load_state_dict(torch.load("model/fraud_gnn.pt"))
    model.eval()
    pathlib.Path("model").mkdir(exist_ok=True)
    torch.onnx.export(model, (g.x, g.edge_index), ONNX_OUT,
                      input_names=["x","edge_index"], output_names=["prob"],
                      opset_version=17)
if __name__ == "__main__":
    export()
