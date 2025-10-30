
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