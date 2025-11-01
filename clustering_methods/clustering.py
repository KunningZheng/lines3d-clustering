
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
    from clustering_methods.bottom_up_merging.merging import init_merging, bottom_up_merging

    all_mask_to_lines3d = init_merging(all_lines3d_to_masks)

    lines_clusters = bottom_up_merging(all_mask_to_lines3d, threshold=0.5)
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