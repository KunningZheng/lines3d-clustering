import os

## 预处理的过程在这里集合
from preprocess.sfm_reader import load_sparse_model
from preprocess.camera_overlap import match_pair
from utils.lines_correspondence import LineCorrespondence
from preprocess.line3dpp_loader import parse_lines3dpp
from preprocess.mask_association import associate_lines2d_to_masks

# load_and_process_data
def load_and_process_data(path_manager, k_near):
    """Load and process all required data for clustering."""
    # Load sparse model
    camerasInfo, points_in_images, points3d_xyz = load_sparse_model(path_manager.sparse_model_path)
    
    # Compute match matrix
    #match_matrix = match_pair(camerasInfo, points_in_images, k_near=k_near)
    
    # Parse 3D line segments
    lines3d, lines3d_to_lines2d, lines2d_in_cam = parse_lines3dpp(path_manager.line3dpp_path)

    # Associate 2D line to masks
    visualize_path = os.path.join(path_manager.intermediate_output_path, 'mask_lines2d')
    associate_lines2d_to_masks(lines3d_to_lines2d, lines2d_in_cam, camerasInfo, 
                                         path_manager.merged_mask_path, visualize_path)

    
    # Associate 3D line to masks through 2D-3D relation

# associate_line3d_to_mask
