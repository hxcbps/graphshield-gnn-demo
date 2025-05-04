import torch
from sklearn.preprocessing import StandardScaler
def preprocess(df):
    scaler = StandardScaler().fit(df.iloc[:, 2:])
    X = scaler.transform(df.iloc[:, 2:]).astype("float32")
    return torch.tensor(X)
