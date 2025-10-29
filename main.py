from preprocess.preprocess import load_and_process_data
from utils.config import get_config, PathManager

# Load configuration
config = get_config()
path_manager = PathManager(config['workspace_path'], config['scene_name'])

# Load and process data
camerasInfo, lines3d, all_lines3d_to_masks = load_and_process_data(path_manager, config['k_near'])

# Apply Clustering
