import torch
from matplotlib import pyplot
import torchvision

# 載入模型與設定設備
model = torch.hub.load('ndahlquist/pytorch-hub-stylegan:0.0.1', 'style_gan', pretrained=True)
model.eval()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)

def generate_faces(fixed_elements_count):
    nb_samples = 16
    # 1. 先生成全新的 16 組 512 維隨機向量
    latents = torch.randn(nb_samples, 512, device=device)
    
    # 2. 將前 fixed_elements_count 個元素設為相同
    # 我們取第一組向量的前 N 個元素，覆蓋到所有 16 組向量上
    if fixed_elements_count > 0:
        latents[:, :fixed_elements_count] = latents[0, :fixed_elements_count]

    with torch.no_grad():
        imgs = model(latents)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0
    
    imgs = imgs.cpu()
    grid = torchvision.utils.make_grid(imgs, nrow=4)
    
    pyplot.figure(figsize=(10, 10))
    pyplot.title(f"Fixed first {fixed_elements_count} elements")
    pyplot.imshow(grid.permute(1, 2, 0).detach().numpy())
    pyplot.axis('off')

    pyplot.savefig(
        f"fixed_{fixed_elements_count}_elements.png",
        dpi=300,
        bbox_inches="tight",
    )
    pyplot.show()

# --- 執行實驗 ---

# 任務 1: 固定前 128 個元素
generate_faces(128)

# 任務 2: 固定前 384 個元素
generate_faces(384)