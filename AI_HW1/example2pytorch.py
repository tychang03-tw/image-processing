import torch
import torch.nn as nn
import torch.optim as optim
#from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

import cv2

with open('example2data.npy', 'rb') as f:
    img = np.load(f)
    data_train = np.load(f)
    data_test = np.load(f)
f.close()

model = nn.Sequential(
    nn.Linear(4, 3),
    nn.Sigmoid(),
    nn.Linear(3, 3),
    nn.Sigmoid()
)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.02)

history = []
for epoch in range(1000):
    np.random.shuffle(data_train)
    x_train = torch.tensor(data_train[:,0:4], dtype=torch.float32)
    y_train = torch.tensor(data_train[:,4:7], dtype=torch.float32)
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    history.append(loss.detach())
    print('Epoch: ' + format(epoch) + ', Loss: ' + format(loss))
    loss.backward()
    optimizer.step()

label_train = torch.max(y_train, dim=1)[1]
label_pred = torch.max(y_pred, dim=1)[1]
acc = (label_train == label_pred)
n_train = float(x_train.shape[0])
print('Training Accuracy: ' + format(sum(acc)/n_train))

x_val = torch.tensor(data_test[:,0:4], dtype=torch.float32)
y_pred = model(x_val)
y_val = torch.tensor(data_test[:,4:7], dtype=torch.float32)
label_val = torch.max(y_val, dim=1)[1]
label_pred = torch.max(y_pred, dim=1)[1]
acc = (label_val == label_pred)
n_val = float(x_val.shape[0])
print('Validation Accuracy: ' + format(sum(acc)/n_val))

img = np.moveaxis(img, 0, -1)
imgclass = model(torch.tensor(img, dtype=torch.float32))
imgclass = imgclass.detach().numpy()
imgclass = (imgclass > 0.5).astype('uint8') * 255
cv2.imwrite('water.png',imgclass[:,:,0])
cv2.imwrite('urban.png',imgclass[:,:,1])
cv2.imwrite('veg.png',imgclass[:,:,2])
cv2.imwrite('color.png',img[:,:,0:3])
#summary(model, (100,4))
for param in model.parameters():
    print(param.data)

plt.plot(history)
plt.show()