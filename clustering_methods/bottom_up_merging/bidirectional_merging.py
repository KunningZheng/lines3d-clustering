def bidirectional_merging(cam_id1, cam_id2, merged_results, threshold):
    # offset，偏移cam2的mask_id，避免冲突
    max_id = max(merged_results[cam_id1])
    merged_results[cam_id2] = {k + max_id+1: v for k, v in merged_results[cam_id2].items()}

    # cam1 --> cam2，用cam2的标签体系更新cam1的标签
    new_cam1 = undirectional_merging(merged_results[cam_id1], merged_results[cam_id2], threshold)
    # cam2 --> new_cam1，用new_cam1的标签体系更新cam2的标签
    new_cam2 = undirectional_merging(merged_results[cam_id2], new_cam1, threshold)

    # test:合并new_cam2中交集大于3的两类

    return new_cam2


def undirectional_merging(view1, view2, threshold):
    
    new_view = view2.copy()
    for mask_id1, lines3d_list1 in view1.items():
        num_lines1 = len(lines3d_list1)
        mask_id2_record = {}

        view2_temp = new_view.copy()
        for mask_id2, lines3d_list2 in view2_temp.items():
            num_lines2 = len(lines3d_list2)
            # mask1和mask2的lines3d交集
            common_lines = lines3d_list1.intersection(lines3d_list2)
            # 计算交集比例
            mask_id2_record[mask_id2] = len(common_lines) / min(num_lines1, num_lines2)
        # 找到最大值
        if len(mask_id2_record) != 0:
            # 筛选出ratio大于阈值的mask_id2
            best_mask_id2 = []
            for mask_id2, ratio in mask_id2_record.items():
                if ratio > threshold:
                    best_mask_id2.append(mask_id2)
            if len(best_mask_id2) != 0:
                # 将所有的best_mask_id2和mask_id1一起合并到第一个best_mask_id2中
                mask_id2 = best_mask_id2[0]
                new_view[mask_id2].update(view1[mask_id1])
                for id in best_mask_id2[1:]:
                    new_view[mask_id2].update(view2_temp[id])
                    if id in new_view:
                        del new_view[id]
            # 没有大于阈值的，在view2没有对应的mask，直接合并到view2中
            else:
                new_view[mask_id1] = view1[mask_id1]                

    
    return new_view