import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
import open3d as o3d


def rasterize_lines(image_shape, lines, show=False):
    """
    将线段栅格化，生成一个与图片同尺寸的 numpy 数组，每个像素存储线段编号。
    
    参数：
    - image_shape: (H, W) 代表输出栅格的高度和宽度
    - lines: 线段列表，每条线段的格式为 (x1, y1, x2, y2)
    - show: 是否显示中间结果（可选，默认为 False）
    
    返回：
    - raster: 2D numpy 数组，与 image_shape 相同，包含线段编号，未被线段覆盖的像素为 -1
    """
    H, W = image_shape
    raster_lines = np.full((H, W), -1, dtype=int)  # 初始化栅格，未被覆盖的像素设为 -1
    
    for idx, (y1, x1, y2, x2) in enumerate(lines):
        # 计算线段的像素点
        rr, cc = line(round(y1), round(x1), round(y2), round(x2))  # skimage.draw.line 返回行列索引（y, x）
        
        # 过滤掉超出范围的点
        valid_idx = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[valid_idx], cc[valid_idx]
        
        # 在栅格上标记线段编号
        raster_lines[rr, cc] = idx

    # 可视化
    if show:
        raster = np.where(raster_lines == -1, 255, 0)
        plt.figure(figsize=(10, 10), dpi=100)
        plt.imshow(raster, cmap='gray')

    return raster_lines


# 可视化聚类结果
def visualize_line_clusters(lines3d, clusters):
    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Line Clusters', width=1200, height=800)
    
    # 为每个聚类生成不同颜色
    n_clusters = len(clusters)
    colors = np.random.rand(n_clusters, 3)
    print(f"Total {n_clusters} lines3d clusters")
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
    vis.add_geometry(coordinate_frame)
    
    # 遍历每个聚类
    for i, (cluster_id, line_data) in enumerate(clusters.items()):
        cluster_color = colors[i][:3]  # 取RGB值
        
        # 创建当前聚类的线段集合
        line_set = o3d.geometry.LineSet()
        points = []
        lines = []
        
        # 添加当前聚类的所有线段
        for idx, line_id in enumerate(line_data):
            x1, y1, z1, x2, y2, z2 = lines3d[line_id]
            points.append([x1, y1, z1])
            points.append([x2, y2, z2])
            lines.append([2*idx, 2*idx+1])  # 连接两个点形成线段
        
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
        line_set.paint_uniform_color(cluster_color)
        
        vis.add_geometry(line_set)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.line_width = 5.0  # 设置线段粗细
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深色背景
    
    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()



def line_to_cylinder(p1, p2, radius=0.02, color=[1, 0, 0]):
    """
    将一条线段转换为圆柱体，可控粗细且精确对齐两端点。
    """
    direction = p2 - p1
    height = np.linalg.norm(direction)
    if height < 1e-8:
        return None

    # 创建圆柱，中心在原点，沿Z轴方向，高度为 height
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=20)
    cylinder.paint_uniform_color(color)

    # 旋转：Z轴 -> direction
    direction /= height
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    if np.linalg.norm(axis) < 1e-8:
        R = np.eye(3)
    else:
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cylinder.rotate(R, center=np.zeros(3))

    # 平移到中点位置（注意 cylinder 默认中心在原点）
    mid_point = (p1 + p2) / 2.0
    cylinder.translate(mid_point)

    return cylinder


def visualize_line_clusters2(lines3d, clusters, colors_option = None, line_radius=0.02):
    """
    使用圆柱可视化聚类线段，可控制粗细。
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Line Clusters', width=1200, height=800)
    
    n_clusters = len(clusters)
    if colors_option == None:
        colors = np.random.rand(n_clusters, 3)
    else:
        colors = colors_option
    print(f"Total {n_clusters} clusters")

    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # 遍历聚类
    for i, (cluster_id, line_data) in enumerate(clusters.items()):
        if colors_option == None:
            color = colors[i]
        else:
            # 从256转成0-1范围
            color = [c/255.0 for c in colors[cluster_id]]
        for line_id in line_data:
            x1, y1, z1, x2, y2, z2 = lines3d[line_id]
            p1 = np.array([x1, y1, z1])
            p2 = np.array([x2, y2, z2])
            cylinder = line_to_cylinder(p1, p2, radius=line_radius, color=color)
            if cylinder is not None:
                vis.add_geometry(cylinder)

    # 渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])
    render_option.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    vis.run()
    vis.destroy_window()



def cull_by_fov(R_cw, T_cw, lines3d, max_angle=np.deg2rad(60)):
    # 相机视线方向
    cam_center = T_cw
    cam_dir = R_cw @ np.array([0, 0, 1])
    # 线段中点
    midpoints = (lines3d[:,:3] + lines3d[:, 3:]) / 2
    # 主点指向线段中点
    dirs = midpoints - cam_center
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    
    cos_theta = np.sum(dirs * cam_dir, axis=1)
    visible_mask = cos_theta > np.cos(max_angle) # (N,1)
 
    
    return visible_mask 


def project_lines3d(cam_dict, lines3d):
    R_cw, T_cw = cam_dict['rotation'], cam_dict['position']
    fx, fy = cam_dict['fx'], cam_dict['fy']
    width = cam_dict['width']
    height = cam_dict['height']
    cx = cam_dict.get('cx', width / 2)
    cy = cam_dict.get('cy', height / 2)
    R_wc = np.transpose(R_cw)

    # 1.所有线段端点
    pts = lines3d.reshape(-1,3).T
    # 2.世界 -> 相机坐标
    pts_cam = R_wc @ (pts - T_cw.reshape(3,1))  # (3,N)

    # 3.投影到像素
    x = pts_cam[0, :] / pts_cam[2, :]
    y = pts_cam[1, :] / pts_cam[2, :]
    u = fx * x + cx
    v = fy * y + cy
    proj = np.stack([u, v], axis=1).reshape(-1, 2, 2) # (N,2,2)


    # 4.检查是否在图像内
    visible_mask = (pts_cam[2, :] > 0).reshape(-1,2).all(axis=1) & \
                   (proj[:,:,0] >= 0).all(axis=1) & (proj[:,:,0] < width).all(axis=1) & \
                   (proj[:,:,1] >= 0).all(axis=1) & (proj[:,:,1] < height).all(axis=1)
    # 筛选proj
    proj = proj[visible_mask].reshape(-1, 4)

    return proj, visible_mask

def determine_line3d_visibility(cam_dict, line3d, proj, points3d_xyz_subset):

    R_cw = cam_dict['rotation']  # Camera -> World
    T_cw = cam_dict['position']
    fx = cam_dict['fx']
    fy = cam_dict['fy']
    width = cam_dict['width']
    height = cam_dict['height']
    cx = cam_dict.get('cx', width / 2)
    cy = cam_dict.get('cy', height / 2)

    R_wc = np.transpose(R_cw)
    
    # 线段端点与中心
    p1, p2 = line3d[:3], line3d[3:]
    center = (p1 + p2) / 2
    # 线段投影
    p1_proj, p2_proj = proj[:2], proj[2:]

    # ===  4.利用稀疏点云遮挡测试  ===
    # 计算线段在图像上的投影长度
    #proj_length = np.linalg.norm(p1_proj - p2_proj)
    #if proj_length < 10:  # 投影太短的线段不考虑
        #return False

        
    # 检查线段中间是否有稀疏点云遮挡
    # 采样线段上的点进行检查
    for alpha in np.linspace(0.1, 0.9, 5):  # 采样5个点
        sample_point = p1 * (1-alpha) + p2 * alpha
        sample_cam = R_wc @ (sample_point - T_cw)
        
        # 投影到图像平面
        sample_proj = np.array([fx * sample_cam[0]/sample_cam[2]+cx, fy * sample_cam[1]/sample_cam[2]+cy])
        
        # 检查该点附近是否有更近的3D点
        for pt in points3d_xyz_subset:
            pt_cam = R_wc @ (pt - T_cw)
            if pt_cam[2] <= 0:  # 点在相机后方
                continue
                
            pt_proj = np.array([fx * pt_cam[0]/pt_cam[2]+cx, fy * pt_cam[1]/pt_cam[2]+cy])
            dist = np.linalg.norm(pt_proj - sample_proj)
            
            # 如果附近有点且深度更小（更近），则认为被遮挡
            if dist < 10 and pt_cam[2] < sample_cam[2] * 0.9:
                return False
    return True


def line_segment_distance(seg1, seg2):
    """
    计算两条3D线段的最小距离
    seg1, seg2: np.array([x1,y1,z1,x2,y2,z2])
    """
    p1, q1 = seg1[:3], seg1[3:]
    p2, q2 = seg2[:3], seg2[3:]

    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2

    a = np.dot(u,u)
    b = np.dot(u,v)
    c = np.dot(v,v)
    d = np.dot(u,w0)
    e = np.dot(v,w0)

    denom = a*c - b*b
    sc, tc = 0.0, 0.0
    if denom < 1e-9:
        # 两条线几乎平行
        sc = 0.0
        tc = d/b if abs(b) > abs(c) else e/c
    else:
        sc = (b*e - c*d) / denom
        tc = (a*e - b*d) / denom

    # 裁剪到[0,1]
    sc = np.clip(sc, 0.0, 1.0)
    tc = np.clip(tc, 0.0, 1.0)

    # 最近点
    closest_point_seg1 = p1 + sc*u
    closest_point_seg2 = p2 + tc*v
    return np.linalg.norm(closest_point_seg1 - closest_point_seg2)


# 可视化聚类结果
def visualize_line3d(lines3d, line3d_ids):
    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Line Clusters', width=1200, height=800)
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
    vis.add_geometry(coordinate_frame)
    
    # 可视化所有线段
    line_set = o3d.geometry.LineSet()
    points = []
    lines = []
    for idx, line_id in enumerate(line3d_ids):
        x1, y1, z1, x2, y2, z2 = lines3d[line_id]
        points.append([x1, y1, z1])
        points.append([x2, y2, z2])
        lines.append([2*idx, 2*idx+1])  # 连接两个点形成线段
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.paint_uniform_color([0, 0, 0])  # 绿色
    vis.add_geometry(line_set)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.line_width = 5.0  # 设置线段粗细
    render_option.background_color = np.array([1, 1, 1])
    
    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()