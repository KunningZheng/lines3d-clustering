from clustering_methods.bottom_up_merging.enhanced_bidirectional_merging import bidirectional_merging
from collections import defaultdict

def init_merging(all_lines3d_to_masks_):
    # 转化all_lines3d_to_masks的格式
    # 后面都是逐视角处理，cam_id在第一位方便处理
    all_lines3d_to_masks = defaultdict(dict)
    for line3d_id, line3d_to_mask in all_lines3d_to_masks_.items():
        for cam_id, mask_id in line3d_to_mask.items():
            all_lines3d_to_masks[cam_id][line3d_id] = mask_id

    # 获取all_mask_to_line3d
    all_mask_to_lines3d = {}
    for cam_id, line3d_to_mask in all_lines3d_to_masks.items():
        mask_to_lines3d = {}
        for line3d_id, mask_id in line3d_to_mask.items():
            if mask_id not in mask_to_lines3d:
                mask_to_lines3d[mask_id] = set()
            mask_to_lines3d[mask_id].add(line3d_id)
        all_mask_to_lines3d[cam_id] = mask_to_lines3d
    
    # 整理mask的序号
    for cam_id, line3d_to_mask in all_mask_to_lines3d.items():
        mask_keys = sorted(line3d_to_mask)
        # 创建新的字典，使用连续的序号
        new_line3d_to_mask = {}
        for new_idx, old_key in enumerate(mask_keys):
            new_line3d_to_mask[new_idx] = line3d_to_mask[old_key]
        # 更新原字典
        all_mask_to_lines3d[cam_id] = new_line3d_to_mask
    return all_mask_to_lines3d

def bottom_up_merging(all_mask_to_lines3d, threshold=0.5):    
    merged_results = all_mask_to_lines3d.copy()
    while len(merged_results) > 1:
        # 获取排序后的键
        cam_ids = sorted(merged_results.keys())
        merged = {}
        new_cam_id = 0

        # 两两合并
        import math
        number = math.floor(len(cam_ids)/2)
        for i in range(0, number):
            cam_id1, cam_id2 = cam_ids[2*i], cam_ids[2*i+1]
            merged[new_cam_id] = bidirectional_merging(cam_id1, cam_id2, merged_results, threshold)
            new_cam_id += 1
        
        # 如果视角数为奇数，保留最后一个视角
        if len(cam_ids) % 2 == 1:
            last_cam_id = cam_ids[-1]
            merged[new_cam_id] = merged_results[last_cam_id]

        # 更新merged_results
        merged_results = merged

        # 整理mask的序号
        for cam_id, line3d_to_mask in merged_results.items():
            mask_keys = sorted(line3d_to_mask)
            # 创建新的字典，使用连续的序号
            new_line3d_to_mask = {}
            for new_idx, old_key in enumerate(mask_keys):
                new_line3d_to_mask[new_idx] = line3d_to_mask[old_key]
            # 更新原字典
            merged_results[cam_id] = new_line3d_to_mask

        print(f"After merging, number of views: {len(merged_results)}")
    return merged_results[0]