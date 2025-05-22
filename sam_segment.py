import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class SAMSegmentation:
    def __init__(self, model_type='vit_b', checkpoint_path='path/to/sam_checkpoint.pth'):
        # 加载 SAM 模型
        self.model_type = model_type
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.eval().cuda()  # 如果有 GPU 可用，移动到 GPU

    def segment(self, rgb_image):
        # 将图像转换为模型输入格式
        input_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).cuda()  # 添加批次维度并移动到 GPU

        # 创建预测器
        predictor = SamPredictor(self.sam)
        predictor.set_image(rgb_image)

        # 设置输入点和标签（例如，前景点）
        input_point = np.array([[rgb_image.shape[1] // 2, rgb_image.shape[0] // 2]])
        input_label = np.array([1])  # 1 表示前景

        # 进行预测
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # 找到分数最高的掩码
        best_mask_index = np.argmax(scores)
        best_mask = masks[best_mask_index]

        return best_mask

    @staticmethod
    def convert_to_instance_mask(binary_mask):
        # 确保输入是二值化的
        binary_mask = (binary_mask <= 0).astype(np.uint8)

        # 使用 connectedComponents 函数
        num_labels, labels = cv2.connectedComponents(binary_mask)

        # labels 是一个与输入掩码大小相同的数组，其中每个连通区域有一个唯一的标签
        return labels

# 使用示例
if __name__ == "__main__":
    # 假设您有一个 RGB 图像
    image_path = 'example/rgb.png'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建 SAMSegmentation 实例
    sam_segmenter = SAMSegmentation(model_type='vit_b', checkpoint_path='example/sam_vit_b_01ec64.pth')

    # 获取分数最高的掩码
    best_mask = sam_segmenter.segment(image_rgb)
    instance_mask = sam_segmenter.convert_to_instance_mask(best_mask)
    # 将 instance_mask 转换为 RGB 图像
    # instance_mask_rgb = cv2.applyColorMap(instance_mask.astype(np.uint8) * 255, cv2.COLORMAP_JET)
    # 显示结果
    import matplotlib.pyplot as plt
    plt.imshow(image_rgb)
    plt.imshow(best_mask, alpha=0.5)
    plt.show()
    cv2.imshow("Best Instance Mask", instance_mask.astype(np.uint8) * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()