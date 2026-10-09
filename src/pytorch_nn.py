"""04_pytorch_nn.py
Simple PyTorch neural network with easy comments.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODELS_DIR = os.path.join("results","models")
os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    X_train = np.load(os.path.join(MODELS_DIR,"X_train.npy"))
    X_test = np.load(os.path.join(MODELS_DIR,"X_test.npy"))
    y_train = np.load(os.path.join(MODELS_DIR,"y_train.npy"))
    y_test = np.load(os.path.join(MODELS_DIR,"y_test.npy"))
    return X_train, X_test, y_train, y_test

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_nn(epochs=10, batch_size=256, lr=1e-3):
    X_train, X_test, y_train, y_test = load_data()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_np = y_test

    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim).to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print("Training neural network...")
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} - Loss {total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        X_test_t = X_test_t.to(DEVICE)
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_pred = (probs >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test_np, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_np, y_pred, average="binary", zero_division=0)
    print("=== PyTorch NN ===")
    print("Accuracy :", f"{acc:.4f}")
    print("Precision:", f"{prec:.4f}")
    print("Recall   :", f"{rec:.4f}")
    print("F1-score :", f"{f1:.4f}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR,"pytorch_nn_model.pth"))
    with open(os.path.join(MODELS_DIR,"nn_metrics.txt"),"w") as f:
        f.write("PyTorch NN metrics\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")

if __name__ == "__main__":
    train_nn()
