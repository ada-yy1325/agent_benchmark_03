import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess
import sys

# ===== 0. NPU 环境检测 =====
try:
    import torch_npu
    print(f"torch_npu version: {torch_npu.__version__}")
except ImportError:
    print("[ERROR] torch_npu 未安装！请运行: pip install torch_npu")
    sys.exit(1)

# ===== 1. 检查 NPU 状态 =====
print(f"PyTorch version: {torch.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU device count: {torch.npu.device_count()}")

if not torch.npu.is_available():
    print("[ERROR] NPU 不可用，请检查昇腾驱动是否正常加载")
    sys.exit(1)

# 打印 NPU 指标（非 nvidia-smi，而是 npu-smi）
try:
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5)
    print("\n===== NPU 状态 =====")
    print(result.stdout if result.returncode == 0 else result.stderr)
except Exception as e:
    print(f"[WARNING] 无法获取 NPU 指标: {e}")

# ===== 2. 设置设备 =====
device = torch.device("npu:0")
print(f"\nUsing device: {device}")
print(f"Device name: {torch.npu.get_device_name(0)}")

# ===== 3. 生成合成数据: y = 2.0 * x + 1.0 + noise =====
np.random.seed(42)
torch.manual_seed(42)

X = np.linspace(-5, 5, 200).reshape(-1, 1).astype(np.float32)
y = 2.0 * X + 1.0 + np.random.randn(*X.shape).astype(np.float32) * 0.5

X_tensor = torch.from_numpy(X).to(device)
y_tensor = torch.from_numpy(y).to(device)

print(f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")

# ===== 4. 定义线性回归模型 =====
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ===== 5. 训练循环 =====
epochs = 500
print(f"\n开始训练 {epochs} 个 epoch...")
print("=" * 50)

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

# ===== 6. 最终结果 =====
print("=" * 50)
w, b = model.linear.weight.item(), model.linear.bias.item()
print(f"\n训练完成！")
print(f"模型: y = {w:.4f} * x + {b:.4f}")
print(f"真实: y = 2.0 * x + 1.0")
print(f"设备: {device}")
print("\n✅ 线性回归训练成功！")