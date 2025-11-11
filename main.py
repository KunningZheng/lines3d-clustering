import json
import os 
from preprocess.preprocess import load_and_process_data
from utils.config import get_config, PathManager
import importlib
from utils.json2dict import json_decode_with_int, convert_sets
from utils.lines_tools import visualize_line_clusters
from utils.masks_tools import visualize_masks_clusters

## Load configuration
config = get_config()
path_manager = PathManager(config['workspace_path'], config['scene_name'])

## Load and process data
camerasInfo, match_matrix, lines3d, all_lines3d_to_masks, points3d_xyz = load_and_process_data(path_manager, config['k_near'])

## Apply Clustering
if config['graph_clustering'] != '':
    lines3d_clusters_path = path_manager.get_lines3d_clusters_path(config['clustering_method']+'_'+config['graph_clustering'])
else:
    lines3d_clusters_path = path_manager.get_lines3d_clusters_path(config['clustering_method'])
masks_clusters_json_path = path_manager.get_masks_clusters_path(config['clustering_method'])
# Clustering Results Exists
if os.path.exists(lines3d_clusters_path):
    print('Loading existing clustering results')
    with open(lines3d_clusters_path, 'r') as f:
        lines3d_clusters = json_decode_with_int(json.load(f))
    if config['clustering_method'] == 'mask_clustering':
        with open(masks_clusters_json_path, 'r') as f:
            masks_clusters = json_decode_with_int(json.load(f))
# Clustering
else:
    print('Start Clustering...')
    # Dynamically import and execute clustering function
    module = importlib.import_module('clustering_methods.clustering')
    clustering_function = getattr(module, config['clustering_method'])
    # Mask Clustering will return both lines3d clusters and masks clusters
    if config['clustering_method'] == 'mask_clustering':
        lines3d_clusters, masks_clusters = clustering_function(all_lines3d_to_masks)
        with open(masks_clusters_json_path, 'w') as f:
            json.dump(convert_sets(masks_clusters), f)
    # Other methods will only return lines3d clusters
    elif config['clustering_method'] == 'lines_clustering':
        lines3d_clusters = clustering_function(all_lines3d_to_masks, config['graph_clustering'])
    elif config['clustering_method'] == 'geometry_clustering':
        lines3d_clusters = clustering_function(points3d_xyz, lines3d)
    elif config['clustering_method'] == 'voting_based_clustering':
        lines3d_clusters = clustering_function(all_lines3d_to_masks, match_matrix)
    else:
        lines3d_clusters = clustering_function(all_lines3d_to_masks)
    with open(lines3d_clusters_path, 'w') as f:
        json.dump(convert_sets(lines3d_clusters), f)

## Visualize Line Clusters
visualize_line_clusters(lines3d, lines3d_clusters)

## Visualize 2D Masks Clusters
if config['clustering_method'] == 'mask_clustering':
    masks_clusters_path = os.path.join(path_manager.intermediate_output_path, 'masks_clusters')
    os.makedirs(masks_clusters_path, exist_ok=True)
    visualize_masks_clusters(masks_clusters, path_manager.merged_mask_path, camerasInfo, masks_clusters_path)
