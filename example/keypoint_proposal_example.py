import sys
import os
import cv2

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将keypoint_proposal.py所在的目录添加到Python路径
sys.path.append(os.path.join(current_dir, '..'))

import numpy as np
import torch
from keypoint_proposal import KeypointProposer

# 配置字典
config = {
    'device': 'cuda',
    'bounds_min': [-0.45, -0.75, 0.698],
    'bounds_max': [0.10, 0.60, 1.2],
    'min_dist_bt_keypoints': 0.06,
    'max_mask_ratio': 0.5,
    'num_candidates_per_mask': 5,
    'seed': 0
}

# 初始化关键点提取器
keypoint_proposer = KeypointProposer(config)

# 示例RGB图像和点云数据
rgb_image = cv2.imread(os.path.join(current_dir, 'rgb.png'))
point_cloud = np.load(os.path.join(current_dir, 'points.npy'))
masks = np.load(os.path.join(current_dir, 'mask.npy'))
# 可视化rgb和masks
cv2.imshow("rgb", rgb_image)
masks_display = (masks * 255).astype(np.uint8)
cv2.imshow("masks", masks_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
# transform to torch
rgb_image = torch.from_numpy(rgb_image).permute(0, 1, 2).float() / 255.0
# point_cloud = torch.from_numpy(point_cloud).float()
masks = torch.from_numpy(masks).long()
# print shape
print(f"rgb_image.shape: {rgb_image.shape}")
print(f"point_cloud.shape: {point_cloud.shape}")
print(f"masks.shape: {masks.shape}")
# 获取关键点
candidate_keypoints, projected_image = keypoint_proposer.get_keypoints(rgb_image, point_cloud, masks)

# 输出结果
print("提取的关键点坐标:")
print(candidate_keypoints)

# 显示带有关键点标记的图像
import cv2
cv2.imshow("Projected Keypoints", projected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()