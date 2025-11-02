import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def project_lines(points3d_xyz, lines3d):
    # ======================
    # 1. 读取点云
    # ======================
    # points3d_xyz: dict, {point3d_id: (x,y,z)}
    points = np.array(list(points3d_xyz.values()))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # ======================
    # 2. 估计地面平面（RANSAC）
    # ======================
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.5,  # 阈值可根据点云精度调整
        ransac_n=3,
        num_iterations=1000
    )
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    print("Plane normal:", normal)

    # ======================
    # 3. 计算旋转矩阵，使平面法向量对齐Z轴
    # ======================
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

    # ======================
    # 4. 投影到地面XY平面
    # ======================
    points_in_lines = lines3d.reshape(-1, 3)
    rotated_points = (R @ points_in_lines.T).T
    projected_lines = rotated_points[:, :2].reshape(-1,4)

    return projected_lines


def visualize_projected_lines(projected_lines, labels):
    """
    可视化投影线段的聚类结果
    """
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    np.random.seed(42)
    colors = np.random.rand(len(unique_labels), 3)

    for i, label in enumerate(unique_labels):
        if label == -1:
            color = 'gray'
            alpha = 0.3
        else:
            color = colors[i]
            alpha = 0.7
        mask = labels == label
        lines = projected_lines[mask]
        for line in lines:
            x1, y1, x2, y2 = line
            plt.plot([x1, x2], [y1, y2], color=color, alpha=alpha)

    plt.title('Projected 2D Lines with HDBSCAN Clustering (Fast KDTree Version)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(False)
    plt.show()