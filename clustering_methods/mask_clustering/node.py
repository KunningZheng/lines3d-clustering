import torch
import open3d as o3d

class Node:
    
    def __init__(self, mask_list, visible_frame, contained_mask, line3d_ids, node_info, son_node_info):
        '''
            mask_list: list of masks that is within this cluster
            visible_frame: one-hot vector, 1 if the node appears in the frame
            contained_mask: one-hot vector, 1 if the node is contained by the mask
            line3d_ids: the corresponding 3D line ids
            node_info: for debugging. The iteration and the index of the node in this iteration
            son_node_info: for debugging. Node infos from the last iteration that are merged into this node
        
        '''
        self.mask_list = mask_list
        self.visible_frame = visible_frame
        self.contained_mask = contained_mask
        self.line3d_ids = line3d_ids
        self.node_info = node_info
        self.son_node_info = son_node_info


    @ staticmethod
    def create_node_from_list(node_list, node_info):
        mask_list = []
        visible_frame = torch.zeros(len(node_list[0].visible_frame), dtype=bool)
        contained_mask = torch.zeros(len(node_list[0].contained_mask), dtype=bool)
        line3d_ids = set()
        son_node_info = set()
        for node in node_list:
            mask_list += node.mask_list
            visible_frame = visible_frame | (node.visible_frame).bool()
            contained_mask = contained_mask | (node.contained_mask).bool()
            line3d_ids = line3d_ids.union(node.line3d_ids)
            son_node_info.add(node.node_info)
        return Node(mask_list, visible_frame.float(), contained_mask.float(), line3d_ids, node_info, son_node_info)

    
    # def get_point_cloud(self, scene_points):
    #     '''
    #         return:
    #             pcld: open3d.geometry.PointCloud object, the point cloud of the node
    #             point_ids: list of int, the corresponding 3D point ids of the node
    #     '''
    #     line3d_ids = list(self.line3d_ids)
    #     points = scene_points[line3d_ids]
    #     pcld = o3d.geometry.PointCloud()
    #     pcld.points = o3d.utility.Vector3dVector(points)
    #     return pcld, line3d_ids
