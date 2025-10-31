import igraph as ig

def leiden_community(edges, weights, num_segments, idx_to_segment):

    # 创建图
    g = ig.Graph()
    g.add_vertices(num_segments)
    g.add_edges(edges)
    g.es['weight'] = weights

    # 使用Leiden算法进行社区发现
    partition = g.community_leiden(
        objective_function='modularity',
        weights='weight',
        resolution_parameter=1.0,
        n_iterations=-1  # 直到收敛
    )
    
    # 收集聚类结果
    clusters = {}
    for idx, label in enumerate(partition.membership):
        seg = idx_to_segment[idx]
        clusters.setdefault(label, set()).add(seg)
    # 剔除小于3个线段的聚类
    clusters = {k: list(v) for k, v in clusters.items() if len(v) >= 3}
    return clusters