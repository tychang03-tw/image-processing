import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# 1. 載入萬用風格轉換模型 (這會自動從網路下載，不需要 .t7 檔)
print("正在載入萬用模型...")
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_img(path_to_img):
    # 1. 改用 OpenCV 讀取 (它可以讀 tiff)
    img = cv2.imread(path_to_img)
    if img is None:
        print(f"錯誤：找不到圖片 {path_to_img}")
        return None
    # 2. OpenCV 是 BGR，要轉成 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. 轉成 TensorFlow 的格式 (0~1 之間的 float32)
    img = img.astype(np.float32) / 255.0
    # 4. 增加維度 (從 H, W, C 變成 1, H, W, C)
    img = img[np.newaxis, :]
    # 5. 轉成 Tensor 物件
    return tf.convert_to_tensor(img)

# 2. 設定檔案路徑
base_path = os.path.dirname(os.path.abspath(__file__))
content_path = os.path.join(base_path, 'F16.tiff')

# 你資料夾裡現有的 6 張風格圖片檔名 (.jpg)
style_files = [
    'la_muse.jpg', 'rain_princess.jpg', 'the_scream.jpg',
    'the_shipwreck_of_the_minotaur.jpg', 'udnie.jpg', 'wave.jpg'
]

# 3. 讀取內容圖
content_image = load_img(content_path)

# 4. 進行轉換並排版
plt.figure(figsize=(16, 10))
plt.subplot(2, 4, 1)
plt.imshow(np.squeeze(content_image))
plt.title("Original (F16)")
plt.axis('off')

print("開始分析圖片風格並轉換...")
for i, style_name in enumerate(style_files):
    style_path = os.path.join(base_path, style_name)
    if not os.path.exists(style_path):
        print(f"找不到風格圖: {style_name}")
        continue
    
    # 讀取風格圖
    style_image = load_img(style_path)
    
    # 讓電腦「讀圖」並轉換 (直接輸入內容圖 + 風格圖)
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    
    # 顯示結果
    plt.subplot(2, 4, i + 2)
    plt.imshow(np.squeeze(stylized_image))
    plt.title(style_name)
    plt.axis('off')
    print(f"完成: {style_name}")

plt.tight_layout()
plt.savefig('style_transfer_final.png')
plt.show()