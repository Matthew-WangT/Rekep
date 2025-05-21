import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from utils import filter_points_by_bounds
from sklearn.cluster import MeanShift

class KeypointProposer:
    """
    关键点提取器类，用于从RGB图像和点云中提取有意义的关键点
    利用DINOv2视觉模型提取特征，并通过聚类方法找出关键点
    """
    def __init__(self, config):
        """
        初始化关键点提取器
        
        参数:
            config: 配置字典，包含设备、边界、聚类参数等
        """
        self.config = config
        self.device = torch.device(self.config['device'])
        # 加载DINOv2视觉模型用于特征提取
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self.device)
        # local_model_path = '/omnigibson-src/ReKep/dinov2_vits14_pretrain.pth'
        # checkpoint = torch.load(local_model_path)
        # self.dinov2 = checkpoint
        # 设置工作空间边界
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        # 初始化Mean Shift聚类器，用于合并空间上靠近的关键点
        self.mean_shift = MeanShift(bandwidth=self.config['min_dist_bt_keypoints'], bin_seeding=True, n_jobs=32)
        self.patch_size = 14  # DINOv2模型的patch大小
        # 设置随机种子以确保结果可复现
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

    def get_keypoints(self, rgb, points, masks):
        """
        主要方法：从输入数据中提取关键点
        
        参数:
            rgb: RGB图像
            points: 3D点云数据
            masks: 分割掩码
            
        返回:
            candidate_keypoints: 提取的关键点坐标
            projected: 关键点可视化结果
        """
        import time
        total_start = time.time()
        
        # 预处理
        preprocess_start = time.time()
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)
        preprocess_time = time.time() - preprocess_start
        
        # 提取图像特征
        feature_start = time.time()
        features_flat = self._get_features(transformed_rgb, shape_info)
        feature_time = time.time() - feature_start
        
        # 对每个掩码区域，在特征空间中聚类以获取有意义的区域，并使用它们的中心作为关键点候选
        cluster_start = time.time()
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(points, features_flat, masks)
        cluster_time = time.time() - cluster_start
        
        # 排除工作空间外的关键点
        filter_start = time.time()
        within_space = filter_points_by_bounds(candidate_keypoints, self.bounds_min, self.bounds_max, strict=True)
        candidate_keypoints = candidate_keypoints[within_space]
        candidate_pixels = candidate_pixels[within_space]
        candidate_rigid_group_ids = candidate_rigid_group_ids[within_space]
        filter_time = time.time() - filter_start
        
        # 通过在笛卡尔空间中聚类来合并靠近的点
        merge_start = time.time()
        merged_indices = self._merge_clusters(candidate_keypoints)
        candidate_keypoints = candidate_keypoints[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]
        merge_time = time.time() - merge_start
        
        # 按位置对候选点进行排序
        sort_start = time.time()
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]
        sort_time = time.time() - sort_start
        
        # 将关键点投影到图像空间进行可视化
        project_start = time.time()
        projected = self._project_keypoints_to_img(rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat)
        project_time = time.time() - project_start
        
        total_time = time.time() - total_start
        
        # 打印时间统计
        print("\n===== 关键点提取时间统计 =====")
        print(f"预处理时间: {preprocess_time*1000:.2f} ms ({preprocess_time/total_time*100:.1f}%)")
        print(f"特征提取时间: {feature_time*1000:.2f} ms ({feature_time/total_time*100:.1f}%)")
        print(f"特征聚类时间: {cluster_time*1000:.2f} ms ({cluster_time/total_time*100:.1f}%)")
        print(f"空间过滤时间: {filter_time*1000:.2f} ms ({filter_time/total_time*100:.1f}%)")
        print(f"点合并时间: {merge_time*1000:.2f} ms ({merge_time/total_time*100:.1f}%)")
        print(f"排序时间: {sort_time*1000:.2f} ms ({sort_time/total_time*100:.1f}%)")
        print(f"投影可视化时间: {project_time*1000:.2f} ms ({project_time/total_time*100:.1f}%)")
        print(f"总耗时: {total_time*1000:.2f} ms")
        print("==============================\n")
        
        return candidate_keypoints, projected

    def _preprocess(self, rgb, points, masks):
        """
        预处理输入数据
        
        参数:
            rgb: RGB图像
            points: 3D点云数据
            masks: 分割掩码
            
        返回:
            transformed_rgb: 调整大小后的RGB图像
            rgb: 原始RGB图像
            points: 3D点云数据
            masks: 转换为二进制掩码列表
            shape_info: 图像形状信息字典
        """
        if masks.is_cuda:
            masks = masks.cpu()
            # print("***masks", masks)
            
        rgb = rgb.cpu()  # 如果在GPU上，则移至CPU
        rgb = rgb.numpy() 
        # print("***rgb", rgb)
            
        # 将掩码转换为二进制掩码列表
        masks = [masks == uid for uid in np.unique(masks.numpy())]
        # print("***masks2", masks)
        # 确保输入形状与DINOv2兼容
        H, W, _ = rgb.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size
        # print("***rgb2", rgb)
        transformed_rgb = cv2.resize(rgb, (new_W, new_H))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # 转换为float32 [H, W, 3]
        # 形状信息
        shape_info = {
            'img_h': H,
            'img_w': W,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
        return transformed_rgb, rgb, points, masks, shape_info
    
    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat):
        """
        将关键点投影到图像上进行可视化
        
        参数:
            rgb: 原始RGB图像
            candidate_pixels: 关键点在图像上的像素坐标
            candidate_rigid_group_ids: 关键点所属的刚体组ID
            masks: 分割掩码
            features_flat: 特征向量
            
        返回:
            projected: 带有关键点标记的可视化图像
        """
        projected = rgb.copy()
        # 在图像上叠加关键点
        for keypoint_count, pixel in enumerate(candidate_pixels):
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)
            # 绘制方框
            box_width = 30 + 10 * (text_length - 1)
            box_height = 30
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
            # 绘制文本
            org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
            color = (255, 0, 0)
            cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            keypoint_count += 1
        return projected

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):
        """
        使用DINOv2模型提取图像特征
        
        参数:
            transformed_rgb: 预处理后的RGB图像
            shape_info: 图像形状信息
            
        返回:
            features_flat: 展平的特征向量 [H*W, feature_dim]
        """
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']
        # 获取特征
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)  # float32 [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        # breakpoint()
        import time
        time_start = time.time()
        features_dict = self.dinov2.forward_features(img_tensors)
        time_end = time.time()
        print(f"DINOv2推理时间: {(time_end - time_start)*1000:.2f} ms")
        raw_feature_grid = features_dict['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w, -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
        
        # 计算特征的平均值或最大值等统计量
        feature_vis = torch.mean(raw_feature_grid, dim=-1).squeeze(0).cpu().numpy()
        # 归一化到0-255范围
        feature_vis = ((feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min()) * 255).astype(np.uint8)
        # 显示
        cv2.imshow("feature_visualization", feature_vis)
        
        # 使用双线性插值计算每个点的特征
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(img_h, img_w),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(0)  # float32 [H, W, feature_dim]
        
        # 可视化双线性插值后的特征
        # 计算特征的平均值作为可视化
        interpolated_feature_vis = torch.mean(interpolated_feature_grid, dim=-1).cpu().numpy()
        # 归一化到0-255范围
        interpolated_feature_vis = ((interpolated_feature_vis - interpolated_feature_vis.min()) / 
                                   (interpolated_feature_vis.max() - interpolated_feature_vis.min()) * 255).astype(np.uint8)
        # 应用彩色映射以便更好地可视化
        interpolated_feature_vis_color = cv2.applyColorMap(interpolated_feature_vis, cv2.COLORMAP_JET)
        # 显示
        cv2.imshow("interpolated_features", interpolated_feature_vis_color)
        cv2.waitKey(1)  # 添加短暂等待，确保图像能显示出来
        
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])  # float32 [H*W, feature_dim]
        return features_flat

    def _cluster_features(self, points, features_flat, masks):
        """
        在特征空间中聚类以找出关键点候选
        
        参数:
            points: 3D点云数据
            features_flat: 展平的特征向量
            masks: 分割掩码列表
            
        返回:
            candidate_keypoints: 候选关键点的3D坐标
            candidate_pixels: 候选关键点在图像上的像素坐标
            candidate_rigid_group_ids: 候选关键点所属的刚体组ID
        """
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []
        for rigid_group_id, binary_mask in enumerate(masks):
            # 忽略过大的掩码
            # print("***binary_mask", binary_mask)
            binary_mask = binary_mask.cpu().numpy()
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue
            # 只考虑前景特征
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]
            # 降维以减少对噪声和纹理的敏感性
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            # 将原始特征投影到前三个主成分上
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])
            
            # 使用OpenCV可视化PCA降维后的特征
            if 0 and features_pca.shape[0] > 0:  # 确保有数据可视化
                # 将特征转换为numpy数组
                pca_features_np = features_pca.cpu().numpy()
                
                # 创建一个空白图像
                vis_size = 400
                vis_img = np.ones((vis_size, vis_size, 3), dtype=np.uint8) * 255
                
                # 将前两个主成分映射到2D图像空间
                for i in range(pca_features_np.shape[0]):
                    x = int(pca_features_np[i, 0] * (vis_size - 20) + 10)
                    y = int(pca_features_np[i, 1] * (vis_size - 20) + 10)
                    # 使用第三个主成分作为颜色
                    color_val = int(pca_features_np[i, 2] * 255)
                    color = (color_val, 255 - color_val, 128)  # BGR格式
                    # 绘制点
                    cv2.circle(vis_img, (x, y), 3, color, -1)
                
                # 添加标题
                cv2.putText(vis_img, f"PCA Features (Mask: {rigid_group_id})", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # 显示图像
                cv2.imshow(f"PCA Features (Mask: {rigid_group_id})", vis_img)
                cv2.waitKey(1)  # 非阻塞显示
            
            X = features_pca
            # 添加空间坐标作为额外维度
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch  = (feature_points_torch - feature_points_torch.min(0)[0]) / (feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([X, feature_points_torch], dim=-1)
            # 聚类特征以获取有意义的区域
            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config['num_candidates_per_mask'],
                distance='euclidean',
                device=self.device,
            )
            cluster_centers = cluster_centers.to(self.device)
            for cluster_id in range(self.config['num_candidates_per_mask']):
                cluster_center = cluster_centers[cluster_id][:3]
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        """
        使用Mean Shift算法合并空间上靠近的关键点

        参数:
            candidate_keypoints: 候选关键点的3D坐标
        返回:
            merged_indices: 合并后的关键点索引
        """
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices