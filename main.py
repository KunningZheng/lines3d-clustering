import json
import os 
from preprocess.preprocess import load_and_process_data
from utils.config import get_config, PathManager
import importlib
from utils.json2dict import json_decode_with_int, convert_sets
from utils.lines_tools import visualize_line_clusters
from utils.masks_tools import visualize_masks_clusters

# Load configuration
config = get_config()
path_manager = PathManager(config['workspace_path'], config['scene_name'])

# Load and process data
camerasInfo, lines3d, all_lines3d_to_masks = load_and_process_data(path_manager, config['k_near'])

# Apply Clustering

lines3d_clusters_path = path_manager.get_lines3d_clusters_path(config['clustering_method'])
masks_clusters_json_path = path_manager.get_masks_clusters_path(config['clustering_method'])
if os.path.exists(lines3d_clusters_path):
    with open(lines3d_clusters_path, 'r') as f:
        loaded = json.load(f)
        lines3d_clusters = json_decode_with_int(loaded)
    '''
    with open(masks_clusters_json_path, 'r') as f:
        loaded = json.load(f)
        masks_clusters = json_decode_with_int(loaded)
    '''
else:
    # 动态导入模块
    module = importlib.import_module('clustering_methods.clustering')
    # 动态获取函数
    clustering_function = getattr(module, config['clustering_method'])
    if config['clustering_method'] == 'mask_clustering':
        # 调用函数进行聚类
        lines3d_clusters, masks_clusters = clustering_function(all_lines3d_to_masks)
        # 存储mask_clusters
        with open(masks_clusters_json_path, 'w') as f:
            json.dump(convert_sets(masks_clusters), f)
    else:
        lines3d_clusters = clustering_function(all_lines3d_to_masks)
    # 存储聚类结果
    lines3d_clusters_path = path_manager.get_lines3d_clusters_path(config['clustering_method'])
    with open(lines3d_clusters_path, 'w') as f:
        json.dump(convert_sets(lines3d_clusters), f)
visualize_line_clusters(lines3d, lines3d_clusters)
# 可视化masks clusters
masks_clusters_path = os.path.join(path_manager.intermediate_output_path, 'masks_clusters')
os.makedirs(masks_clusters_path, exist_ok=True)
visualize_masks_clusters(masks_clusters, path_manager.merged_mask_path, camerasInfo, masks_clusters_path)