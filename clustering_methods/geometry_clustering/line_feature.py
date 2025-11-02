import numpy as np
import numpy as np
from tqdm import tqdm
import hdbscan
from sklearn.neighbors import KDTree
import scipy.sparse

def line_features(lines):
    """
    提取线段特征: 中点、方向、长度
    """
    mid = (lines[:, :2] + lines[:, 2:]) / 2
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    length = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    feats = np.column_stack([mid, np.cos(angle), np.sin(angle)])
    return feats


def calculate_projection_distance(line1, line2):
    """
    计算两条线段之间的投影距离
    line1, line2: [x1,y1,x2,y2]
    """
    def point_to_line_distance(point, line):
        x, y = point
        x1, y1, x2, y2 = line
        line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if line_length == 0:
            return np.sqrt((x-x1)**2 + (y-y1)**2)
        t = ((x-x1)*(x2-x1) + (y-y1)*(y2-y1)) / (line_length**2)
        t = max(0, min(1, t))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        return np.sqrt((x-proj_x)**2 + (y-proj_y)**2)
    
    p1, p2 = line1[:2], line1[2:]
    p3, p4 = line2[:2], line2[2:]
    d1 = point_to_line_distance(p1, line2)
    d2 = point_to_line_distance(p2, line2)
    d3 = point_to_line_distance(p3, line1)
    d4 = point_to_line_distance(p4, line1)
    return min(d1, d2, d3, d4)

def cluster_lines_hdbscan_projection(projected_lines, min_cluster_size=5, min_samples=None, search_radius=10.0):
    """
    使用KDTree加速的HDBSCAN聚类
    参数:
        projected_lines: (N,4)
        min_cluster_size: HDBSCAN参数
        min_samples: HDBSCAN参数
        search_radius: 限制只在这个范围内计算线段间距离（单位同坐标）
    """
    n_lines = projected_lines.shape[0]
    midpoints = (projected_lines[:, :2] + projected_lines[:, 2:]) / 2.0

    # 建立 KDTree
    tree = KDTree(midpoints)

    rows, cols, vals = [], [], []

    print(f"Building sparse distance matrix (radius={search_radius}) ...")
    for i in tqdm(range(n_lines), desc="Neighbor distance calc"):
        neighbors = tree.query_radius(midpoints[i:i+1], r=search_radius)[0]
        for j in neighbors:
            if j <= i:
                continue
            dist = calculate_projection_distance(projected_lines[i], projected_lines[j])
            rows.append(i)
            cols.append(j)
            vals.append(dist)

    # 构造稀疏对称矩阵
    sparse_mat = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n_lines, n_lines))
    sparse_mat = sparse_mat + sparse_mat.T

    if min_samples is None:
        min_samples = min_cluster_size

    print("Running HDBSCAN on sparse precomputed matrix ...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='precomputed')
    labels = clusterer.fit_predict(sparse_mat)

    return labels