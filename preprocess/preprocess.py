import os
import json

## 预处理的过程在这里集合
from preprocess.sfm_reader import load_sparse_model
from preprocess.camera_overlap import match_pair
from preprocess.line3dpp_loader import parse_lines3dpp
from preprocess.mask_association import associate_lines2d_to_masks, associate_lines3d_to_masks


def convert_keys_to_int(obj):
    if isinstance(obj, dict):
        return {int(k): convert_keys_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_int(item) for item in obj]
    return obj


# load_and_process_data
def load_and_process_data(path_manager, k_near):
    """Load and process all required data for clustering."""
    # Load sparse model
    camerasInfo, points_in_images, points3d_xyz = load_sparse_model(path_manager.sparse_model_path)
    
    # Compute match matrix
    #match_matrix = match_pair(camerasInfo, points_in_images, k_near=k_near)
    
    # Parse 3D line segments
    lines3d, lines3d_to_lines2d, lines2d_in_cam = parse_lines3dpp(path_manager.line3dpp_path)

    # Associate 3D Lines to Masks
    all_lines3d_to_masks_path = os.path.join(path_manager.intermediate_output_path, 'all_lines3d_to_masks.json')
    if os.path.exists(all_lines3d_to_masks_path):
        # Read existing results
        print('Loading existing lines3d and masks asssociation file')
        with open(all_lines3d_to_masks_path, 'r') as f:
            all_lines3d_to_masks = json.load(f)
        all_lines3d_to_masks = convert_keys_to_int(all_lines3d_to_masks)
    
    else:
        # Associate 2D line to masks
        visualize_path = os.path.join(path_manager.intermediate_output_path, 'mask_lines2d')
        lines2d_in_cam = associate_lines2d_to_masks(lines2d_in_cam, camerasInfo, 
                                            path_manager.merged_mask_path, output_path=None)

        # Associate 3D line to masks through 2D-3D relation
        all_lines3d_to_masks = associate_lines3d_to_masks(lines3d_to_lines2d, lines2d_in_cam)
        
        # Save Results in json
        with open(all_lines3d_to_masks_path, 'w') as f:
            json.dump(all_lines3d_to_masks, f, indent=4)

    return camerasInfo, lines3d, all_lines3d_to_masks


