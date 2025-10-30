import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

# 可视化同类mask
def visualize_masks_clusters(clusters, merged_mask_path, camerasInfo, output_path):
    
    # 为每个mask ID生成唯一颜色
    color_dict = {}
    used_colors = {tuple([0, 0, 0])}  # 禁止使用背景色
    
    for mask_id in range(len(clusters)):
        while True:
            # 生成随机RGB颜色
            color = tuple([random.randint(0, 255) for _ in range(3)])
            
            # 检查颜色是否唯一且不与背景色接近
            color_diff = np.abs(np.array(color) - [0, 0, 0])
            if color not in used_colors and np.all(color_diff > 50):
                used_colors.add(color)
                color_dict[mask_id] = color
                break

    # 字典：mask对应cluster_id
    mask_to_cluster = defaultdict(dict)
    for i, cluster in clusters.items():
        for (cam_id, mask_id) in cluster:
            mask_to_cluster[cam_id][mask_id] = i
    
    for cam_id in tqdm(range(len(camerasInfo)),desc='Visualize mask clusters:'):
        cam_dict = camerasInfo[cam_id]
        img_name = cam_dict['img_name']
        W,H = int(cam_dict['width']), int(cam_dict['height'])

        # 读取当前航片下的merged_mask
        mask_path = os.path.join(merged_mask_path, img_name + '_maskraw.png')
        merged_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        merged_mask = merged_mask.astype(np.int16) - 1  # 转回原始ID，背景为-1
        
        # 创建黑色背景 [0,0,0]
        img = np.full((H, W, 3), [0, 0, 0], dtype=np.uint8)
        for mask_id, cluster_id in mask_to_cluster[cam_id].items():
            # 创建当前mask的布尔掩码
            mask_area = (merged_mask == mask_id)
            # 用对应颜色填充mask区域
            img[mask_area] = color_dict[cluster_id]

        # 存储赋色图
        filename = f"{img_name}_mask_cluster.png"
        cv2.imwrite(os.path.join(output_path, filename), img)        

