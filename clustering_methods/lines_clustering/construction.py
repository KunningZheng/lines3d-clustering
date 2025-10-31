import numpy as np
import time
import logging
from scipy.sparse import csr_matrix

def build_similarity_graph(all_lines3d_to_masks, required_views=3, sim_threshold=0.1):
    """
    构建加权图边表
    """
    # === 1. 收集视图和掩码 ===
    num_segments = len(all_lines3d_to_masks)
    views = sorted({cam_id for m in all_lines3d_to_masks.values() for cam_id in m})
    masks = sorted({(cam_id, mask_id) for m in all_lines3d_to_masks.values() for cam_id, mask_id in m.items()})
    view_to_idx = {v: i for i, v in enumerate(views)}
    mask_to_idx = {m: i for i, m in enumerate(masks)}

    # === 2. 构造稀疏矩阵 ===
    view_rows, view_cols = [], []
    mask_rows, mask_cols = [], []
    for i, (_, mapping) in enumerate(all_lines3d_to_masks.items()):
        for cam_id, mask_id in mapping.items():
            view_rows.append(i)
            view_cols.append(view_to_idx[cam_id])
            mask_rows.append(i)
            mask_cols.append(mask_to_idx[(cam_id, mask_id)])

    view_matrix = csr_matrix(
        (np.ones(len(view_rows)), (view_rows, view_cols)),
        shape=(num_segments, len(views)), dtype=np.int32)
    mask_matrix = csr_matrix(
        (np.ones(len(mask_rows)), (mask_rows, mask_cols)),
        shape=(num_segments, len(masks)), dtype=np.int32)

    # === 3. 稀疏矩阵乘法计算共视与共mask ===
    common_views = (view_matrix @ view_matrix.T).tocsr()
    common_masks = (mask_matrix @ mask_matrix.T).tocsr()

    # === 4. 提取有效边（只保留共视>阈值 且相似度>阈值）===
    edges = []
    weights = []
    common_views = common_views.tocoo()
    for i, j, v in zip(common_views.row, common_views.col, common_views.data):
        if i < j and v > required_views:
            sim = common_masks[i, j] / v
            if sim > sim_threshold:
                edges.append((i, j))
                weights.append(float(sim))

    print(f"构建图完成: {len(edges)} 条边，{num_segments} 个节点")
    return edges, weights, num_segments

