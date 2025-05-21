import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
from segment_anything import sam_model_registry, SamPredictor

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load image
image = cv2.imread(os.path.join(current_dir, 'rgb.png'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM model
sam_checkpoint = os.path.join(current_dir, 'sam_vit_b_01ec64.pth')
model_type = "vit_b"
# sam_checkpoint = os.path.join(current_dir, 'sam_vit_h_4b8939.pth')
# model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.eval().cuda()  # Move to GPU if available

# 记录开始时间
start_time = time.time()

predictor = SamPredictor(sam)
predictor.set_image(image)

# Set input point and label (e.g., foreground point at x=500, y=400)
input_point = np.array([[500, 400]])
input_label = np.array([1])  # 1 for foreground, 0 for background

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # returns multiple masks
)
time_cost = time.time() - start_time
print(f"Time cost: {time_cost} seconds")
# Show masks
for i, mask in enumerate(masks):
    plt.figure()
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    plt.title(f"Mask {i+1} (Score: {scores[i]:.3f})")
    plt.axis('off')
    plt.show()
