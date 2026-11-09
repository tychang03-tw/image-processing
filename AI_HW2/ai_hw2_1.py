import torch
from torch import nn
import torchvision
import torchvision.transforms as T
from collections import OrderedDict

# https://github.com/mateuszbuda/brain-segmentation-pytorch

def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=64):
        super(UNet, self).__init__()

        features = init_features
        # 修改為 2 層結構 (原本是 3 層)
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck 層
        self.bottleneck = UNet._block(features * 2, features * 4, name="bottleneck")

        # 對應的 2 層 Decoder
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # Forward pass 也要同步縮減層數
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1) # Skip connection
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1) # Skip connection
        dec1 = self.decoder1(dec1)
        
        # 注意：訓練 Oxford Pets 通常輸出 3 類別，最後不建議用 sigmoid，
        # 因為 CrossEntropyLoss 會內含 Softmax
        return self.conv(dec1) 

    @staticmethod
    def _block(in_channels, features, name):
        # 將 conv2 修改為 Grouped Convolution (groups=features)
        # 注意：in_channels 必須能被 groups 整除
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            groups=features, # <--- 這裡修改為分組卷積
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

def train_loop(model, loader, epochs, device):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epoch_i, epoch_j = epochs
    for i in range(epoch_i, epoch_j):
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        running_loss = 0.0
        running_samples = 0
        for batch_idx, (inputs, targets) in enumerate(loader, 0):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze(dim=1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_samples += targets.size(0)
            running_loss += loss.item()

            print("Trained {} samples, Loss: {:.4f}".format(running_samples, running_loss / (batch_idx+1)))

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print('device:', device)

transforms = T.Compose([T.ToTensor(), T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST)])
target_transforms = T.Compose([T.ToTensor(), T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST), T.Lambda(tensor_trimap)])
pets_train = torchvision.datasets.OxfordIIITPet(root="datasets/OxfordPets/train/", split="trainval", target_types="segmentation",
                         download=True, transform=transforms, target_transform=target_transforms)
pets_train_loader = torch.utils.data.DataLoader(pets_train, batch_size=64, shuffle=True)

model = UNet()
model = model.to(device)
model.train()

train_loop(model, pets_train_loader, (1, 101), device)

torch.save(model.state_dict(), 'ai_hw2_1.pth')