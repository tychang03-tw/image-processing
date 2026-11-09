import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 準備數據 (保持不變)
x_train = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
# CrossEntropyLoss 在 PyTorch 中通常使用類別索引 (0 或 1)
# 這裡 y_train 改為類別索引：[1,0]->0, [1,0]->0, [0,1]->1, [0,1]->1
y_train = torch.tensor([0, 0, 1, 1], dtype=torch.long) 

# 2. 定義 4 層神經網路
class FourLayerNet(nn.Module):
    def __init__(self, hidden_size=8):
        super(FourLayerNet, self).__init__()
        # 層 1: 輸入(2) -> 隱藏1
        self.fc1 = nn.Linear(2, hidden_size)
        # 層 2: 隱藏1 -> 隱藏2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 層 3: 隱藏2 -> 隱藏3
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # 層 4: 隱藏3 -> 輸出(2類別)
        self.fc4 = nn.Linear(hidden_size, 2)
        
        self.relu = nn.ReLU() # 隱藏層改用 ReLU 通常效果更好

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x) # 注意：CrossEntropyLoss 內部自帶 Softmax，最後一層不加激活
        return x

model = FourLayerNet(hidden_size=16)

# 3. 設定損失函數為 Cross Entropy
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam 通常比 SGD 跑得快

# 4. 訓練模型
history_loss = []
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    history_loss.append(loss.item())

# 5. 測試與繪圖
grid = np.linspace(-1.5, 1.5, 31)
xx, yy = np.meshgrid(grid, grid)
x_val_torch = torch.from_numpy(np.stack([xx, yy], axis=-1).reshape(-1, 2)).float()

model.eval()
with torch.no_grad():
    logits = model(x_val_torch)
    # 使用 Softmax 取得機率值供視覺化
    probs = torch.softmax(logits, dim=1)

y_val_np = probs.numpy().reshape(31, 31, 2)

# 繪圖
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_wireframe(xx, yy, y_val_np[:, :, 0], color='blue')
ax1.set_title("Probability of Class 0")

ax2 = fig.add_subplot(122)
ax2.plot(history_loss)
ax2.set_title("Cross Entropy Loss")
plt.show()