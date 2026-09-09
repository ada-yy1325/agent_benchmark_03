import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 检查 NPU
device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 生成合成数据: y = 2.0 * x + 1.0 + noise
np.random.seed(42)
torch.manual_seed(42)

X = np.linspace(-5, 5, 200).reshape(-1, 1).astype(np.float32)
y = 2.0 * X + 1.0 + np.random.randn(*X.shape).astype(np.float32) * 0.5

X_tensor = torch.from_numpy(X).to(device)
y_tensor = torch.from_numpy(y).to(device)

print(f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")

# 2. 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. 训练
epochs = 500
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        w, b = model.linear.weight.item(), model.linear.bias.item()
        print(f"Epoch [{epoch:3d}/{epochs}], Loss: {loss.item():.6f}, w: {w:.4f}, b: {b:.4f}")

# 4. 结果
w, b = model.linear.weight.item(), model.linear.bias.item()
print(f"\nFinal: y = {w:.4f} * x + {b:.4f}")
print(f"Ground truth: y = 2.0 * x + 1.0")
print(f"Training completed on {device}")