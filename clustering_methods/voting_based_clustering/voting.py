from collections import defaultdict
from collections import deque

def voting_by_masks(all_lines3d_to_masks, match_matrix):
    # 转换格式
    all_masks_to_lines3d_ = defaultdict(lambda: defaultdict(set))
    for line3d_id, masks in all_lines3d_to_masks.items():
        for cam_id, mask_id in masks.items():
            all_masks_to_lines3d_[cam_id][mask_id].add(line3d_id)
    all_masks_to_lines3d = dict(all_masks_to_lines3d_.copy())
    # 初始化跨视角的掩码关联
    all_masks_association = {}
    
    ####################### 每张航片循环 #######################
    for cam_id, masks_to_lines3d in all_masks_to_lines3d.items():
        # 查找临近航片
        neighbor_ids = match_matrix[cam_id][1:]
        
        ####################### 每个mask循环 #######################
        for mask_id, lines3d_ids in masks_to_lines3d.items():
            # 初始化投票向量
            neighbor_votes=[]
            neighbor_masks=[]
            ####################### 每张临近航片循环 #######################
            for neighbor_id in neighbor_ids:
                neighbor_id = int(neighbor_id)  # 确保邻近航片ID是整数类型
                ### step1:查找同名线段，并记录其对应的mask_id ###
                correspond_masks = []
                for line3d_id in lines3d_ids:
                    neighbor_mask_id = all_lines3d_to_masks[line3d_id].get(neighbor_id, None)
                    if neighbor_mask_id is not None:
                        correspond_masks.append(neighbor_mask_id)
                ### step2:根据同名线段的mask数量投票 ###
                if correspond_masks != []:
                    unique_masks = set(correspond_masks)
                    vote = len(unique_masks)
                    if vote == 1:
                        neighbor_masks.append(correspond_masks[0])  # 记录唯一的mask
                    else:
                        # 如果有多个mask，则记录所有的mask
                        neighbor_masks.extend(unique_masks)
                    neighbor_votes.append(vote)
                    # vote=0，能见部分少
                    # vote=1，在邻近航片中有唯一mask
                    # vote>1，在邻近航片中有多个mask
                else:
                    neighbor_votes.append(0)  # 没有找到对应的mask
                    neighbor_masks.append(-1)  # -1表示没有对应的mask

            ### step3：判定当前mask是否是correct mask ###
            # 如果是false mask，则跳过
            if max(neighbor_votes) != 1:
                continue
            # 如果是correct mask
            else:
                ####################### 每张临近航片循环 #######################
                for idx in range(len(neighbor_ids)):
                    # 如果vote=1，则为有效投票，建立跨视角的掩码关联
                    if neighbor_votes[idx] == 1:
                        neighbor_id = int(neighbor_ids[idx])
                        # 记录当前mask和邻近航片的关联
                        all_masks_association.setdefault((cam_id, mask_id), []).append((neighbor_id, neighbor_masks[idx]))
                        all_masks_association.setdefault((neighbor_id, neighbor_masks[idx]), []).append((cam_id, mask_id))       
    
    return all_masks_association, all_masks_to_lines3d


def masks_clustering(all_masks_association):
    ### 统计视角间关联的实例 ###
    # 创建节点集合
    all_nodes = set(all_masks_association.keys())
    visited = set()
    masks_clusters = {}
    cluster_id = 0

    # 使用BFS遍历图查找连通分量
    for node in all_nodes:
        if node not in visited:
            # 初始化新簇
            current_cluster = []
            queue = deque([node])
            visited.add(node)
            
            while queue:
                current_node = queue.popleft()
                current_cluster.append(current_node)
                
                # 遍历所有邻居
                for neighbor in all_masks_association.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # 存储当前簇
            masks_clusters[cluster_id] = current_cluster
            cluster_id += 1

    # 打印聚类结果
    print(f"Found {len(masks_clusters)} instance clusters")
    return masks_clusters


def lines_clustering(masks_clusters, all_masks_to_lines3d):
    # 聚类3D线段
    lines3d_clusters = {}
    for cluster_id, masks in masks_clusters.items():
        for mask in masks:
            cam_id, mask_id = mask
            lines3d_ids = all_masks_to_lines3d[cam_id][mask_id]
            for line3d_id in lines3d_ids:
                lines3d_clusters.setdefault(cluster_id, set()).add(line3d_id)
    return lines3d_clusters