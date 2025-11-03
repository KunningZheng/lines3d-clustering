
def mask_clustering(all_lines3d_to_masks):
    from clustering_methods.mask_clustering.construction import mask_graph_construction
    from clustering_methods.mask_clustering.iterative_clustering import iterative_clustering
    
    # Construct Graph
    nodes, observer_num_thresholds = mask_graph_construction(all_lines3d_to_masks)
    # Graph Clustering
    nodes = iterative_clustering(nodes, observer_num_thresholds, connect_threshold=0.9, debug=True)
    # Output lines Clusters
    lines_clusters = {}
    for i, node in enumerate(nodes):
        lines_clusters[i] = list(node.line3d_ids)
    # Output Masks Clusters
    masks_clusters = {}
    for i, node in enumerate(nodes):
        masks_clusters[i] = list(node.mask_list)    
    return lines_clusters, masks_clusters


def bottom_up_merging(all_lines3d_to_masks):
    from clustering_methods.bottom_up_merging.merging import init_merging, bottom_up_merging_cam

    all_mask_to_lines3d = init_merging(all_lines3d_to_masks)

    lines_clusters = bottom_up_merging_cam(all_mask_to_lines3d, threshold=0.5)
    return lines_clusters


def lines_clustering(all_lines3d_to_masks, graph_clustering):
    from clustering_methods.lines_clustering.construction import build_similarity_graph
    # 收集所有线段索引
    all_segments = sorted(all_lines3d_to_masks.keys())
    
    # 创建线段索引到连续整数的映射
    segment_to_idx = {seg: idx for idx, seg in enumerate(all_segments)}
    idx_to_segment = {idx: seg for seg, idx in segment_to_idx.items()}

    # 计算相似度矩阵
    edges, weights, num_segments = build_similarity_graph(all_lines3d_to_masks, required_views=3, sim_threshold=0.1)

    # 聚类
    if graph_clustering == 'leiden_community':
        from clustering_methods.lines_clustering.graph_clustering import leiden_community
        line_clusters = leiden_community(edges, weights, num_segments, idx_to_segment)
    elif graph_clustering == 'chinese_whispers':
        from clustering_methods.lines_clustering.graph_clustering import chinese_whispers
        line_clusters = chinese_whispers(edges, weights, num_segments, idx_to_segment)
    elif graph_clustering == 'node2vec_hdbscan':
        from clustering_methods.lines_clustering.graph_clustering import node2vec_hdbscan
        line_clusters = node2vec_hdbscan(edges, weights, num_segments, idx_to_segment)
    elif graph_clustering == 'direct_hdbscan':
        from clustering_methods.lines_clustering.graph_clustering import direct_hdbscan
        line_clusters = direct_hdbscan(edges, weights, num_segments, idx_to_segment) 
    return line_clusters


def geometry_clustering(points3d_xyz, lines3d):
    from clustering_methods.geometry_clustering.project_lines import project_lines
    # 1. 估计地面平面并投影线段
    projected_lines = project_lines(points3d_xyz, lines3d)

    # 2. 计算线段之间的距离并聚类
    from clustering_methods.geometry_clustering.line_feature import line_features
    features = line_features(projected_lines)
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    labels = clusterer.fit_predict(features)

    #from clustering_methods.geometry_clustering.line_feature import cluster_lines_hdbscan_projection
    #labels = cluster_lines_hdbscan_projection(projected_lines)

    # 3. 可视化结果
    from clustering_methods.geometry_clustering.project_lines import visualize_projected_lines
    visualize_projected_lines(projected_lines, labels)

    # 4. 输出聚类结果
    lines_clusters = {}
    for idx, label in enumerate(labels):
        if label not in lines_clusters:
            lines_clusters[int(label)] = []
        lines_clusters[int(label)].append(int(idx))
    # 剔除小于3个线段的聚类
    lines_clusters = {k: list(v) for k, v in lines_clusters.items() if len(v) >= 3}
    # 剔除label=-1的噪声点聚类
    if -1 in lines_clusters:
        del lines_clusters[-1]
    return lines_clusters