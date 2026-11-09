import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. 讀取數據
with open('example2data.npy', 'rb') as f:
    img = np.load(f)
    data_train = np.load(f)
    data_test = np.load(f)

# 2. 定義 4 層神經網路 (輸入層不算，共有 4 個 Linear 層)
# 假設隱藏層節點為 16 (你可以自由調整)
model = nn.Sequential(
    nn.Linear(4, 16),   # 第 1 層
    nn.ReLU(),
    nn.Linear(16, 16),  # 第 2 層
    nn.ReLU(),
    nn.Linear(16, 8),   # 第 3 層
    nn.ReLU(),
    nn.Linear(8, 3)     # 第 4 層 (輸出層，3 個類別)
)

# 3. 修改損失函數為 CrossEntropyLoss
# 注意：CrossEntropyLoss 的標籤 y 必須是類別索引 (0, 1, 2)，而不是 One-hot
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

history = []

# 準備訓練標籤：從 One-hot 轉成 類別索引
y_train_indices = torch.tensor(np.argmax(data_train[:, 4:7], axis=1), dtype=torch.long)
x_train = torch.tensor(data_train[:, 0:4], dtype=torch.float32)

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred_logits = model(x_train)
    
    loss = criterion(y_pred_logits, y_train_indices)
    loss.backward()
    optimizer.step()
    
    history.append(loss.item())
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

# 4. 驗證與預測
model.eval()
with torch.no_grad():
    # 處理影像分類
    img_reshaped = np.moveaxis(img, 0, -1)
    img_tensor = torch.tensor(img_reshaped, dtype=torch.float32)
    
    # 預測機率 (對最後一維做 Softmax)
    output_logits = model(img_tensor)
    probs = torch.softmax(output_logits, dim=-1).numpy()

# 5. 儲存結果 (使用 0.5 作為閥值)
imgclass = (probs > 0.5).astype('uint8') * 255
cv2.imwrite('water.png', imgclass[:,:,0])
cv2.imwrite('urban.png', imgclass[:,:,1])
cv2.imwrite('veg.png', imgclass[:,:,2])

plt.plot(history)
plt.title("Cross Entropy Loss")
plt.show()