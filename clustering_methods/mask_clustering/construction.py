import numpy as np
from collections import defaultdict
import torch
from clustering_methods.mask_clustering.node import Node


def mask_graph_construction(all_lines3d_to_masks):
    # Prepare for computing rate
    masks_to_lines3d, visible_frames, contained_masks = associate_mask_to_frame(all_lines3d_to_masks)
    # Get observer_num_thredshold
    observer_num_thresholds = get_observer_num_thresholds(visible_frames)
    # Initialize Clustering
    nodes = init_nodes(masks_to_lines3d, visible_frames, contained_masks)
    return nodes, observer_num_thresholds


def associate_mask_to_frame(all_lines3d_to_masks_, vis_thred=0.3, contain_thred=0.8):
    '''
    计算各mask在其他视角的可见性和对应的masks，为计算View Consensus Rate做准备
    - Params
        - all_line3d_to_mask_: 3D lines to masks across all views
        - vis_thred: define a mask to be visible at the frame if at least vis_thred of masks' total lines are visible
        - contain_thred: define the approximate containment relationship
    - Return
        - all_mask_to_frame: key是2D Mask(cam_id, mask_id), value是对应的mask的字典, 其中key是cam_id, value是mask_id
    '''
    # 转换all_line3d_to_mask的格式
    masks_to_lines3d = defaultdict(set)
    lines3d_to_frames = defaultdict(set)
    all_lines3d_to_masks = defaultdict(dict)
    for line3d_id, line3d_to_mask in all_lines3d_to_masks_.items():
        for cam_id, mask_id in line3d_to_mask.items():
            masks_to_lines3d[(cam_id, mask_id)].add(line3d_id)
            lines3d_to_frames[line3d_id].add(cam_id)
            all_lines3d_to_masks[cam_id][line3d_id] = mask_id
    
    # 设定global_id和2D mask之间的对应关系
    global_id_to_mask = {}
    mask_to_global_id = {}
    for global_mask_id, (cam_id, mask_id) in enumerate(masks_to_lines3d):
        global_id_to_mask[global_mask_id] = (cam_id, mask_id)
        mask_to_global_id[(cam_id, mask_id)] = global_mask_id

    # 计算可见2DMask的frame以及在该frame中对应的Mask
    print('Associate mask to frame...')
    all_mask_to_frame = {}
    for mask, line3d_ids in masks_to_lines3d.items():
        mask_to_frame = {}
        num_lines3d = len(line3d_ids)

        # 2DMask可能可见的frames
        possible_frames = []
        possible_lines3d = []
        for line3d_id in line3d_ids:
            frames = lines3d_to_frames[line3d_id]  # 可见line3d的frames
            possible_frames.extend(list(frames))
            possible_lines3d += [line3d_id] * len(frames)  # possible_frames对应的lines3d
        # 根据阈值选择可见的frames
        cam_ids, cam_counts = np.unique(possible_frames, return_counts=True)
        select_vis = cam_counts > vis_thred * num_lines3d
        if len(select_vis) == 0:  # 没有其他可见的frames
            all_mask_to_frame[mask] = None
            continue
        else:  # 选择可见的frame
            vis_frames = cam_ids[select_vis]
        
        # 可见的frames中的可见lines3d以及对应的mask
        for cam_id in vis_frames:
            cam_id = int(cam_id)
            # 可见的lines3d
            vis_lines3d = [line for line, frame in zip(possible_lines3d, possible_frames) if frame == cam_id]
            # 对应的masks
            vis_masks = []
            for line3d_id in vis_lines3d:
                vis_masks.append(all_lines3d_to_masks[cam_id][line3d_id])
            # 根据阈值判断从属的masks
            vis_mask_ids, vis_mask_counts = np.unique(vis_masks, return_counts=True)
            if len(vis_mask_counts > contain_thred * len(vis_lines3d)) == 0:  # 没有从属的masks
                mask_to_frame[cam_id] = None  
                continue
            else:
                mask_to_frame[cam_id] = int(vis_mask_ids[np.argmax(vis_mask_counts)])  # 可能有多个超过阈值的，取最大的
            
        all_mask_to_frame[mask] = mask_to_frame   
    
    frame_num = len(all_lines3d_to_masks)
    visible_frames = []
    contained_masks = []
    for global_mask_id, (cam_id, mask_id) in global_id_to_mask.items(): 
        # 获取记录可见framd的one-hot vector
        frame_ids = sorted(all_mask_to_frame[(cam_id, mask_id)].keys())
        frame = torch.zeros(frame_num)
        frame[frame_ids] = 1
        visible_frames.append(frame)
        # 获取记录包含mask的one-hot vector
        frame_mask = torch.zeros(len(global_id_to_mask))
        for cam_id, mask_id in all_mask_to_frame[(cam_id, mask_id)].items():
            global_id = mask_to_global_id[(cam_id, mask_id)]
            frame_mask[global_id] = 1
            contained_masks.append(frame_mask)
    visible_frames = torch.stack(visible_frames, dim=0)  # (mask_num, frame_num)
    contained_masks = torch.stack(contained_masks, dim=0)  # (mask_num, mask_num)

    return masks_to_lines3d, visible_frames, contained_masks


def init_nodes(masks_to_lines3d, visible_frames, contained_mask):
    nodes = []
    for global_mask_id, (cam_id, mask_id) in enumerate(masks_to_lines3d):
        # 当前node中包含的2D mask
        mask_list = [(cam_id, mask_id)]    
        # 获取记录可见framd的one-hot vector
        frame = visible_frames[global_mask_id]
        # 获取记录包含mask的one-hot vector
        frame_mask = contained_mask[global_mask_id]
        # 获取2D mask包含的line3d id(集合)
        line3d_ids = masks_to_lines3d[(cam_id, mask_id)]
        node_info = (0, len(nodes))
        node = Node(mask_list, frame, frame_mask, line3d_ids, node_info, None)
        nodes.append(node)
    return nodes


def get_observer_num_thresholds(visible_frames):
    '''
        Compute the observer number thresholds for each iteration. Range from 95% to 0%.
    '''
    observer_num_matrix = torch.matmul(visible_frames, visible_frames.transpose(0,1))
    observer_num_list = observer_num_matrix.flatten()
    observer_num_list = observer_num_list[observer_num_list > 0].cpu().numpy()
    observer_num_thresholds = []
    for percentile in range(95, -5, -5):
        observer_num = np.percentile(observer_num_list, percentile)
        if observer_num <= 1:
            if percentile < 50:
                break
            else:
                observer_num = 1
        observer_num_thresholds.append(observer_num)
    return observer_num_thresholds