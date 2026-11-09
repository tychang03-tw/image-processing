import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. Data
# ----------------------------
X = torch.tensor([[1,1], [-1,1], [-1,-1], [1,-1],
                  [4,4], [-4,4], [-4,-4], [4,-4]], dtype=torch.float32)
y = torch.tensor([0,0,0,0, 1,1,1,1], dtype=torch.long)

# ----------------------------
# 2. Network
# ----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16) 
        self.fc2 = nn.Linear(16, 2)
        self.act = nn.LeakyReLU(0.1)  

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# ----------------------------
# 3. Training
# ----------------------------
loss_list = []
for epoch in range(300):
    optimizer.zero_grad()
    out = net(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss = {loss.item():.6f}")

# ----------------------------
# 4. Test
# ----------------------------
probs = net(X)
pred = torch.argmax(probs, dim=1)
print("Predicted:", pred.tolist())
print("True:     ", y.tolist())

# ----------------------------
# 5. Plot Loss Curve
# ----------------------------
plt.figure(figsize=(6,4))
plt.plot(loss_list, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------
# 6. Plot Decision Boundary (2D)
# ----------------------------
xx, yy = torch.meshgrid(torch.linspace(-10,10,300), torch.linspace(-10,10,300), indexing='ij')
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

with torch.no_grad():
    Z = net(grid)
    Z = torch.argmax(Z, dim=1).numpy()
    Z = Z.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z, levels=[-0.5,0.5,1.5], colors=['lightblue','lightcoral'], alpha=0.5)
X_np = X.numpy()
plt.scatter(X_np[:4,0], X_np[:4,1], c='blue', s=80, label='Class 0')
plt.scatter(X_np[4:,0], X_np[4:,1], c='red', s=80, label='Class 1')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Neural Network Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 7. Plot 3D Decision Surface
# ----------------------------
from mpl_toolkits.mplot3d import Axes3D

with torch.no_grad():
    probs = torch.softmax(net(grid), dim=1)[:,1].numpy()
    Z_prob = probs.reshape(xx.shape)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx.numpy(), yy.numpy(), Z_prob, cmap='coolwarm', alpha=0.7)
ax.scatter(X_np[:4,0], X_np[:4,1], np.zeros(4), c='blue', s=80, label='Class 0')
ax.scatter(X_np[4:,0], X_np[4:,1], np.ones(4), c='red', s=80, label='Class 1')

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Probability(Class 1)")
ax.set_title("3D Decision Surface")
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.view_init(elev=30, azim=120)
ax.legend()
plt.show()
