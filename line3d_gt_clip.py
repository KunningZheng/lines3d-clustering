import json
import numpy as np
import os
import sys
from preprocess.line3dpp_loader import parse_lines3dpp
import open3d as o3d

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon

# ---------------------------
# 辅助函数
# ---------------------------
def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def point_in_poly_2d(x, y, poly):
    """
    2D 点在多边形内检测（射线法 / even-odd rule）
    poly: list of (x,y) vertices, polygon may be non-convex
    返回 True 如果点在多边形内或在边上
    """
    num = len(poly)
    inside = False
    for i in range(num):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % num]
        # check if point is exactly on the segment
        # compute bounding box first
        if min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12 and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12:
            # cross product to check collinearity
            dx1 = x2 - x1
            dy1 = y2 - y1
            dx2 = x - x1
            dy2 = y - y1
            cross = dx1 * dy2 - dy1 * dx2
            if abs(cross) < 1e-9:
                return True  # on edge
        # ray casting
        intersect = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1)
        if intersect:
            inside = not inside
    return inside


def ortho_project_point(pt, cam_center, forward, right, up):
    """
    CloudCompare 的正交投影：把点投影到以 viewpoint 为原点，基向量为 (right, up) 的坐标系。
    返回 2D 坐标 (x, y) 和 z 深度（可忽略）。
    """
    v = pt - cam_center
    x = np.dot(v, right)
    y = np.dot(v, up)
    z = np.dot(v, forward)  # 深度，可用于front/back判断
    
    return (x, y, z)


def visualize_line_segments(line_segments, vertices=None):
    """
    使用Open3D可视化线段
    """
    # 创建LineSet对象
    line_set = o3d.geometry.LineSet()
    
    # 提取所有唯一的点
    all_points = []
    line_indices = []
    point_index_map = {}
    
    for i, segment in enumerate(line_segments):
        start_point = segment[:3]
        end_point = segment[3:]
        
        # 检查点是否已存在
        start_key = tuple(start_point)
        end_key = tuple(end_point)
        
        if start_key not in point_index_map:
            point_index_map[start_key] = len(all_points)
            all_points.append(start_point)
        
        if end_key not in point_index_map:
            point_index_map[end_key] = len(all_points)
            all_points.append(end_point)
        
        line_indices.append([point_index_map[start_key], point_index_map[end_key]])
    
    # 设置点和线
    line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(line_indices))
    
    # 设置线段颜色（红色）
    colors = [[1, 0, 0] for _ in range(len(line_indices))]  # 红色线段
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # 如果提供了原始顶点，也显示出来
    geometries = [line_set]
    
    if vertices is not None:
        # 创建点云显示顶点
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(vertices))
        point_cloud.paint_uniform_color([0, 1, 0])  # 绿色顶点
        geometries.append(point_cloud)
    
    # 可视化
    o3d.visualization.draw_geometries(geometries, window_name="OBJ线段可视化")

def visualize_projected_segments_on_plane(
    segments,
    poly2d,
    viewpoint,
    forward,
    right,
    up,
    focal_distance=1.0,
    figsize=(8, 8),
    show=True,
    save_path=None,
    plot_all=False
):
    """
    在成像平面上可视化投影后的线段与 polyline（2D）。
    - segments: ndarray (N,6) or list of [x1,y1,z1,x2,y2,z2]
    - poly2d: list/ndarray of (x,y) 顶点（已经在同一平面基底 right/up 里坐标化）
              （如果你传入的是 world-space poly verts，请先把它们投影到平面得到 poly2d）
    - viewpoint, forward, right, up, focal_distance: 与 project_point_to_image_plane 使用相同
    - plot_all: 若 True，会尝试将投影不可见（某端在相机后方）的线段也用虚线/标记显示（仅当一端或两端中的点能投影时）
    返回 (fig, ax)
    """

    # 确保 numpy 数组
    segs = np.asarray(segments, dtype=float)
    N = segs.shape[0]

    # 存储投影好的线段（两端都可投影并在多边形内）
    lines_in = []
    endpoints_in = []

    # （可选）记录那些至少有一个端点可投影但不满足在内的线段（用于绘制虚线）
    lines_partial = []
    lines_outside = []

    for i in range(N):
        p1 = segs[i, 0:3]
        p2 = segs[i, 3:6]
        x1, y1, z1 = ortho_project_point(p1, viewpoint, forward, right, up)
        x2, y2, z2 = ortho_project_point(p2, viewpoint, forward, right, up)

        if z1 > 0 and z2 > 0:
            # 判断是否在多边形内（点在边上也视为内）
            inside1 = point_in_poly_2d(x1, y1, poly2d)
            inside2 = point_in_poly_2d(x2, y2, poly2d)
            if inside1 and inside2:
                lines_in.append([(x1, y1), (x2, y2)])
                endpoints_in.append((x1, y1))
                endpoints_in.append((x2, y2))
            else:
                lines_outside.append([(x1, y1), (x2, y2)])

    # 开始绘图
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal', adjustable='box')

    # 绘制 poly2d 多边形（边界 + 半透明填充）
    try:
        poly_patch = MplPolygon(poly2d, closed=True, fill=True, alpha=0.15, edgecolor='k', linewidth=1.0)
        ax.add_patch(poly_patch)
        # 边界线（更显眼）
        poly_edge = MplPolygon(poly2d, closed=True, fill=False, edgecolor='k', linewidth=1.2)
        ax.add_patch(poly_edge)
    except Exception:
        # 如果 poly2d 非法，则忽略
        pass

    # 绘制内部线段（蓝色实线）
    if len(lines_in) > 0:
        lc_in = LineCollection(lines_in, linewidths=1.2)
        ax.add_collection(lc_in)
        # 标出端点
        pts = np.array(endpoints_in)
        ax.scatter(pts[:, 0], pts[:, 1], s=6)

    # 绘制多余/在外的线段（灰色虚线）
    if len(lines_outside) > 0:
        lc_out = LineCollection(lines_outside, linestyles='dashed', linewidths=0.8, alpha=0.7)
        ax.add_collection(lc_out)

    # 绘制部分可见的线段（红色虚线）
    if len(lines_partial) > 0:
        lc_part = LineCollection(lines_partial, linestyles='dashdot', linewidths=0.9, alpha=0.9)
        ax.add_collection(lc_part)

    # 自动缩放到所有绘制元素
    all_x = []
    all_y = []
    for polyxy in poly2d:
        all_x.append(polyxy[0]); all_y.append(polyxy[1])
    for L in lines_in + lines_outside + lines_partial:
        for (x,y) in L:
            all_x.append(x); all_y.append(y)
    if len(all_x) > 0:
        margin_x = (max(all_x) - min(all_x)) * 0.05 if max(all_x) != min(all_x) else 1.0
        margin_y = (max(all_y) - min(all_y)) * 0.05 if max(all_y) != min(all_y) else 1.0
        ax.set_xlim(min(all_x)-margin_x, max(all_x)+margin_x)
        ax.set_ylim(min(all_y)-margin_y, max(all_y)+margin_y)

    ax.set_xlabel("plane X (right)")
    ax.set_ylabel("plane Y (up)")
    ax.set_title("Projected segments and polyline on imaging plane")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=200)

    if show:
        plt.show()

    return fig, ax


# ---------------------------
# 主流程
# ---------------------------
def main(poly_json_path, line3d_path, output_json):
    # 1) 读取 polyline JSON
    if not os.path.exists(poly_json_path):
        print(f"找不到 {poly_json_path}，请确认路径。")
        sys.exit(1)
    poly_data = json.load(open(poly_json_path, "r"))

    # 2) 读取 segments
    segments, _, _ = parse_lines3dpp(line3d_path)
    N = segments.shape[0]
    P1 = segments[:, 0:3]
    P2 = segments[:, 3:6]

    # 3) 对每个 polyline 处理
    results = {}
    count = 0
    for poly in poly_data:
        if poly == None:
            continue
        name = count
        verts = np.array(poly.get("vertices", []), dtype=float)
        viewpoint = poly.get("viewpoint", None)
        view_normal = poly.get("view_normal", None)
        up_dir = poly.get("up_dir", None)
        focal_distance = poly.get("focal_distance", None)

        viewpoint = np.array(viewpoint, dtype=float)
        forward = -normalize(view_normal)

        # 计算 up, right 基底
        up = normalize(up_dir)
        right = normalize(np.cross(forward, up))

        # 将 polyline 的 verts 投影到成像平面（得到 2D polygon）
        plane_center = viewpoint + forward * focal_distance

        poly2d = []
        # if verts are empty, skip
        if verts.shape[0] == 0:
            results[name] = []
            print(f"{name}: 没有顶点，跳过（0 segments）。")
            continue

        for v in verts:
            # 也可能已经在平面上；无论如何，把它投影到平面并取平面坐标：
            vx, vy, vz = ortho_project_point(v, viewpoint, forward, right, up)
            poly2d.append((vx, vy))

        # 若 poly2d 不形成有效多边形（<3点），则跳过
        if len(poly2d) < 3:
            results[name] = []
            print(f"{name}: 顶点数 < 3，跳过（0 segments）。")
            continue
        '''
        visualize_projected_segments_on_plane(
            segments=segments,
            poly2d=poly2d,
            viewpoint=viewpoint,
            forward=forward,
            right=right,
            up=up,
            focal_distance=focal_distance,
            plot_all=True
        )
        '''
        # 对所有线段批量判断：将每端点投影到 plane 的 2D 坐标
        in_mask = np.zeros(N, dtype=bool)

        # 为速度，向量化前做循环（点投影含 branch，难以完美向量化）
        for i in range(N):
            p1 = P1[i]
            p2 = P2[i]

            x1, y1, z1 = ortho_project_point(p1, viewpoint, forward, right, up)
            if z1 <= 0:
                continue
            x2, y2, z2 = ortho_project_point(p2, viewpoint, forward, right, up)
            if z2 <= 0:
                continue

            # 两端点的投影都在平面上，判断是否在多边形内（包含边）
            inside1 = point_in_poly_2d(x1, y1, poly2d)
            if not inside1:
                continue
            inside2 = point_in_poly_2d(x2, y2, poly2d)
            if not inside2:
                continue

            # 两端点都在内，标记
            in_mask[i] = True

        indices = np.where(in_mask)[0].tolist()
        results[name] = indices
        #visualize_line_segments(segments[indices])

        print(f"{name}: 区域内线段数量 = {len(indices)}")
        count += 1
        
    
    # 保存结果
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    # 可视化线段聚类
    from utils.lines_tools import visualize_line_clusters
    visualize_line_clusters(segments, results)


if __name__ == "__main__":
    poly_json_path = "/home/rylynn/Documents/CloudCompare_Exports/seg_polyline_data.json"
    line3d_path = "/home/rylynn/Pictures/Clustering_Workspace/Shanghai_Region5/Line3D++_LSD/"
    output_json = "polyline_segment_map.json"
    main(poly_json_path, line3d_path, output_json)
