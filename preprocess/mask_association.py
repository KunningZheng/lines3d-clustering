import cv2
import os
import numpy as np
from skimage.draw import line
import random
from tqdm import tqdm

def associate_lines2d_to_masks(lines2d_in_cam, camerasInfo, merged_mask_path, output_path=None):
    '''
    - Params
        - lines2d_in_cam: dict，{cam_id：{seg_id:{'coord': (x1,y1,x2,y2), 'mask_id': -1}}}
        - camerasInfo: dict, colmap_loader.py
        - merged_mask_path: 
        - output_path:
    - Return
        - lines2d_in_cam: update lines2d's mask information
    '''
    for cam_id, lines2d in tqdm(lines2d_in_cam.items(), desc='Associating lines2d to masks'):
        cam_dict = camerasInfo[cam_id]
        img_name = cam_dict['img_name']
        W,H = int(cam_dict['width']), int(cam_dict['height'])

        # 读取当前航片下的merged_mask
        mask_path = os.path.join(merged_mask_path, img_name + '_maskraw.png')
        merged_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        merged_mask = merged_mask.astype(np.int16) - 1  # 转回原始ID，背景为-1

        for seg_id, line2d in lines2d.items():
            x1, y1, x2, y2 = line2d['coord']
            # 计算线段的像素点
            rr, cc = line(round(y1), round(x1), round(y2), round(x2))  # skimage.draw.line 返回行列索引（y, x）
            # 过滤掉超出范围的点
            valid_idx = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            rr, cc = rr[valid_idx], cc[valid_idx]
            # 统计像素点对应的mask_id
            mask_ids, counts = np.unique(merged_mask[rr, cc], return_counts=True)
            # 如果数量最多的ID大于50%，则认为属于该ID，否则仍保持-1的mask
            if np.max(counts) > len(rr) * 0.5:
                best_mask_id = mask_ids[np.argmax(counts)]
                lines2d_in_cam[cam_id][seg_id]['mask'] = best_mask_id
        
        
        # 可视化线段和mask的关联关系
        if output_path:
            merged_mask_resized = cv2.resize(merged_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            visualize_img = visualize_masked_lines((H, W), lines2d, merged_mask_resized)
            filename = f"{img_name}_mask_association.png"
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(visualize_img, cv2.COLOR_RGB2BGR))        
    
    return lines2d_in_cam   


def associate_lines3d_to_masks(lines3d_to_lines2d, lines2d_in_cam):
    '''
    根据line3d和line2d的对应关系，将line3d与mask关联
    - Params
        - lines3d_to_lines2d: dict，{line3d_id: [(cam_id, seg_id)..]}
        - lines2d_in_cam: dict，{cam_id：{seg_id:{'coord': (x1,y1,x2,y2), 'mask_id': -1}}}
    - Returns
        - all_lines3d_to_masks: dict, {line3d_id: {cam_id: mask_id}}
    '''
    all_lines3d_to_masks ={}
    print('Associating lines3d to masks...')
    for line3d_id, lines2d_ids in lines3d_to_lines2d.items():
        line3d_to_masks = {}
        for (cam_id, seg_id) in lines2d_ids:
            mask_id = lines2d_in_cam[cam_id][seg_id]['mask']
            line3d_to_masks[cam_id] = int(mask_id)
        all_lines3d_to_masks[line3d_id] = line3d_to_masks
    return all_lines3d_to_masks


def visualize_masked_lines(image_shape, lines2d, merged_mask):
    """
    可视化线段及其对应的mask区域（半透明填充+线段）
    - Params
        - image_shape: 图像尺寸 (height, width)
        - lines2d: 字典, {seg_id: {coord: [x1,y1,x2,y2], mask: mask_id}}
        - merged_mask: 合并后的mask数组，形状为(height, width)
    - Returns
        - 可视化图像 (RGB格式)
    """
    height, width = image_shape
    # 创建浅灰色背景 [200,200,200]
    img = np.full((height, width, 3), [200, 200, 200], dtype=np.uint8)
    
    # 获取所有有效mask ID
    unique_mask_ids = list(np.unique(merged_mask))
    # 去除背景ID (-1)
    if -1 in unique_mask_ids:
        unique_mask_ids.remove(-1)
    
    # 为每个mask ID生成唯一颜色
    color_dict = {}
    mask_alpha = 0.3  # mask填充透明度
    used_colors = {tuple([200, 200, 200])}  # 禁止使用背景色
    
    for mask_id in unique_mask_ids:
        while True:
            # 生成随机颜色 (BGR格式)
            color = [random.randint(0, 255) for _ in range(3)]
            # 转换为RGB用于颜色检查
            color_rgb = tuple(color[::-1])
            
            # 检查颜色是否唯一且不与背景色接近
            color_diff = np.abs(np.array(color) - [200, 200, 200])
            if color_rgb not in used_colors and np.all(color_diff > 50):
                used_colors.add(color_rgb)
                color_dict[mask_id] = color  # 存储BGR格式颜色
                break
    
    # 先绘制mask区域（半透明填充）
    mask_layer = img.copy()
    for mask_id in color_dict.keys():
        # 创建当前mask的布尔掩码
        mask_area = (merged_mask == mask_id)
        # 用对应颜色填充mask区域
        mask_layer[mask_area] = color_dict[mask_id]
    # 将mask层与原图混合（alpha混合）
    img = cv2.addWeighted(img, 1 - mask_alpha, mask_layer, mask_alpha, 0)
    
    # 再绘制所有线段（不透明）
    for seg_id, seg_data in lines2d.items():
        coord = seg_data['coord']  # [x1, y1, x2, y2]
        mask_id = seg_data['mask']
        if mask_id not in color_dict:
            continue  # 跳过未分配mask
        
        # 注意坐标顺序转换 (x,y格式)
        pt1 = (int(round(coord[0])), int(round(coord[1])))  # (x1, y1)
        pt2 = (int(round(coord[2])), int(round(coord[3])))  # (x2, y2)
        cv2.line(img, pt1, pt2, color_dict[mask_id], thickness=2)
    
    return img