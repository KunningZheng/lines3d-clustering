import os
import json

## 预处理的过程在这里集合
from preprocess.sfm_reader import load_sparse_model
from preprocess.camera_overlap import match_pair
from preprocess.line3dpp_loader import parse_lines3dpp
from preprocess.mask_association import associate_lines2d_to_masks, associate_lines3d_to_masks


def convert_keys_to_int(obj):
    if isinstance(obj, dict):
        return {int(k): convert_keys_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_int(item) for item in obj]
    return obj


# load_and_process_data
def load_and_process_data(path_manager, k_near):
    """Load and process all required data for clustering."""
    # Load sparse model
    camerasInfo, points_in_images, points3d_xyz = load_sparse_model(path_manager.sparse_model_path)
    
    # Compute match matrix
    #match_matrix = match_pair(camerasInfo, points_in_images, k_near=k_near)
    
    # Parse 3D line segments
    lines3d, lines3d_to_lines2d, lines2d_in_cam = parse_lines3dpp(path_manager.line3dpp_path)

    # Associate 3D Lines to Masks
    all_lines3d_to_masks_path = os.path.join(path_manager.intermediate_output_path, 'all_lines3d_to_masks.json')
    if os.path.exists(all_lines3d_to_masks_path):
        # Read existing results
        print('Loading existing lines3d and masks asssociation file')
        with open(all_lines3d_to_masks_path, 'r') as f:
            all_lines3d_to_masks = json.load(f)
        all_lines3d_to_masks = convert_keys_to_int(all_lines3d_to_masks)
    
    else:
        # Associate 2D line to masks
        visualize_path = os.path.join(path_manager.intermediate_output_path, 'mask_lines2d')
        lines2d_in_cam, mask_color_dict = associate_lines2d_to_masks(lines2d_in_cam, camerasInfo, 
                                            path_manager.merged_mask_path, path_manager.images_path, output_path=visualize_path)

        # Associate 3D line to masks through 2D-3D relation
        all_lines3d_to_masks = associate_lines3d_to_masks(lines3d_to_lines2d, lines2d_in_cam)
        
        '''
        # DEBUG: 可视化各视角的line3d
        from utils.lines_tools import visualize_line_clusters2
        from collections import defaultdict
        all_lines3d_to_masks2 = defaultdict(lambda: defaultdict(set))
        for line3d_id, line3d_to_mask in all_lines3d_to_masks.items():
            for cam_id, mask_id in line3d_to_mask.items():
                all_lines3d_to_masks2[cam_id][mask_id].add(line3d_id)
        lines3d_to_mask = all_lines3d_to_masks2[49]
        visualize_line_clusters2(lines3d, lines3d_to_mask, colors_option=mask_color_dict[49], line_radius=0.01)
        lines3d_to_mask = all_lines3d_to_masks2[50]
        visualize_line_clusters2(lines3d, lines3d_to_mask, colors_option=mask_color_dict[50], line_radius=0.01)
        lines3d_to_mask = all_lines3d_to_masks2[51]
        visualize_line_clusters2(lines3d, lines3d_to_mask, colors_option=mask_color_dict[51], line_radius=0.01)
        '''
        # Save Results in json
        with open(all_lines3d_to_masks_path, 'w') as f:
            json.dump(all_lines3d_to_masks, f, indent=4)

    '''
    # TEMP: GT Generation
    # Associate 2D line to masks
    lines3d_clusters_gt_path = os.path.join(path_manager.intermediate_output_path, 'lines3d_clusters_gt.json')
    visualize_path = os.path.join(path_manager.intermediate_output_path, 'mask_lines2d_gt')
    gt_lines2d_in_cam = associate_lines2d_to_masks(lines2d_in_cam, camerasInfo, 
                                        path_manager.gt_mask_path, output_path=visualize_path)

    # Associate 3D line to masks through 2D-3D relation
    gt_all_lines3d_to_masks = associate_lines3d_to_masks(lines3d_to_lines2d, gt_lines2d_in_cam)
    # 直接形成clusters
    from collections import defaultdict, Counter

    lines3d_clusters_gt = defaultdict(set)  # 最终的 cluster 结果（mask_id -> line3d_id 集合）
    line3d_mask_count = defaultdict(Counter)  # 记录每个 line3d 的 mask_id 出现次数：{line3d_id: {mask_id: 出现次数}}

    # 第一步：统计每个 line3d 对应的所有 mask_id 及其出现次数
    for line3d_id, mask_dict in gt_all_lines3d_to_masks.items():
        # mask_dict 是 {cam_id: mask_id}，统计当前 line3d 在所有相机下的 mask_id 重复次数
        for cam_id, mask_id in mask_dict.items():
            line3d_mask_count[line3d_id][mask_id] += 1  # 累计 mask_id 出现次数

    # 第二步：为每个 line3d 选择出现次数最多的 mask_id，构建最终 cluster
    multi_cluster_warnings = []  # 记录被修正的多 cluster line3d（用于日志）

    for line3d_id, mask_counter in line3d_mask_count.items():
        # 筛选出现次数最多的 mask_id（key=mask_id, value=次数）
        # most_common(1) 返回 [(mask_id, 次数)]，若多个 mask_id 次数相同，取首个
        best_mask_id, max_count = mask_counter.most_common(1)[0]
        
        # 检查是否存在多 cluster 情况（用于日志提示）
        if len(mask_counter) > 1:
            # 收集所有关联的 mask_id 及次数，方便排查
            all_mask_info = [(mid, cnt) for mid, cnt in mask_counter.items()]
            multi_cluster_warnings.append({
                "line3d_id": line3d_id,
                "all_mask_info": all_mask_info,
                "selected_mask_id": best_mask_id,
                "selected_count": max_count
            })
        
        # 将 line3d 加入最终的 cluster
        lines3d_clusters_gt[best_mask_id].add(line3d_id)

    # 第三步：输出日志（可选，用于确认修正情况）
    if multi_cluster_warnings:
        print(f"提示：共发现 {len(multi_cluster_warnings)} 个 line3d 关联多个 cluster，已自动修正为出现次数最多的 mask_id：")
        for warning in multi_cluster_warnings:
            line3d_id = warning["line3d_id"]
            all_mask_info = warning["all_mask_info"]
            selected_mask = warning["selected_mask_id"]
            selected_cnt = warning["selected_count"]
            print(f"  line3d_id {line3d_id}：关联 mask_id 及次数 {all_mask_info} → 选择 mask_id {selected_mask}（出现 {selected_cnt} 次）")
    else:
        print("所有 line3d 均只关联一个 mask_id，无需修正。")

    # （可选）若需要将 defaultdict 转为普通 dict（避免后续潜在问题）
    lines3d_clusters_gt = dict(lines3d_clusters_gt)
    from utils.json2dict import convert_sets
    lines3d_clusters_gt = convert_sets(lines3d_clusters_gt)

    # Save Results in json
    from utils.lines_tools import visualize_line_clusters
    with open(lines3d_clusters_gt_path, 'w') as f:
        json.dump(lines3d_clusters_gt, f, indent=4)    
    visualize_line_clusters(lines3d, lines3d_clusters_gt)
    '''
    return camerasInfo, lines3d, all_lines3d_to_masks, points3d_xyz


