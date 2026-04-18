import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Synthetic dataset: y = 3x1 + 2x2 - x3 + noise
def make_dataset(n=1000):
    X = torch.randn(n, 3)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.1 * torch.randn(n)
    return X, y.unsqueeze(1)

class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)

def train():
    torch.manual_seed(42)
    device = torch.device("cpu")

    X, y = make_dataset()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LinearNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print(f"Training on: {device}")
    print(f"Dataset: {len(dataset)} samples, 3 features -> 1 target")
    print(f"True weights: [3.0, 2.0, -1.0], bias: 0.0\n")

    for epoch in range(1, 51):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        if epoch % 10 == 0:
            avg_loss = total_loss / len(dataset)
            w = model.fc.weight.data[0].tolist()
            b = model.fc.bias.data[0].item()
            print(f"Epoch {epoch:3d} | loss: {avg_loss:.6f} | "
                  f"weights: [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}] | bias: {b:.3f}")

    print("\nDone. Learned weights should be close to [3.0, 2.0, -1.0].")

if __name__ == "__main__":
    train()
