import json
import numpy as np
import matplotlib.pyplot as plt
from preprocess.line3dpp_loader import parse_lines3dpp
from utils.json2dict import json_decode_with_int
from utils.config import get_config_eval, PathManager

##--------------- Evaluation Parameters ---------------##
OVERLAPS = np.append(np.arange(0.5, 0.95, 0.05), 0.25) # IoU阈值
MIN_REGION_SIZES = np.array([100])  # 实例的最小线数
DISTANCE_THRESHES = np.array([float('inf')])  # 距离阈值
DISTANCE_CONFS = np.array([-float('inf')])  # 距离置信度


def get_instances(lines3d_clusters_gt, lines3d):
    """
    从字典中提取实例信息
    
    参数:
        lines3d_clusters_gt: 字典，key为实例ID，value为该实例包含的3D线段ID列表
        lines3d: numpy数组，形状为(N, 6)，包含所有3D线段的坐标信息
    
    返回:
        instances: 字典，key为实例ID，value为实例信息字典
    """
    instances = {}
    for instance_id, line_ids in lines3d_clusters_gt.items():
        line_ids = np.array(line_ids)
        lines_count = len(line_ids)
        if lines_count == 0:
            continue
        # 计算质心
        centroid = np.mean(lines3d[line_ids], axis=0)
        # 计算每条线段到质心的距离
        dists = np.linalg.norm(lines3d[line_ids] - centroid, axis=1)
        med_dist = np.median(dists)
        # 简单地将距离置信度设为1.0（可以根据需要调整）
        dist_conf = 1.0
        
        instances[instance_id] = {
            'instance_id': instance_id,
            'lines_count': lines_count,
            'lines_ids': line_ids.tolist(),
            'med_dist': med_dist,
            'dist_conf': dist_conf,
            'matched_pred': []  # 初始化为空列表
        }
    return instances



##--------------- Stage1: 实例匹配，将预测实例与真实实例进行匹配 ---------------##
def assign_instances(gt_file, lines3d, pred_file):
    # 加载真实标注数据
    with open(gt_file, 'r') as f:
        lines3d_clusters_gt = json.load(f)
    # convert keys to int
    lines3d_clusters_gt = json_decode_with_int(lines3d_clusters_gt)
    # 单独提取背景 (-1 标签)
    void_gt_ids = lines3d_clusters_gt.get(-1, [])
    void_gt_ids = np.array(void_gt_ids) if len(void_gt_ids) > 0 else np.array([])
    # 构建实例前剔除背景
    del(lines3d_clusters_gt[-1])
    # 提取真实实例信息
    gt_instances = get_instances(lines3d_clusters_gt, lines3d)

    # 加载预测数据
    with open(pred_file, 'r') as f:
        lines3d_clusters_pred = json.load(f)
    # convert keys to int
    lines3d_clusters_pred = json_decode_with_int(lines3d_clusters_pred)
    # 提取预测实例信息
    pred_instances = get_instances(lines3d_clusters_pred, lines3d)

    # 初始化匹配数据结构
    gt2pred = gt_instances.copy()
    pred2gt = {}
    num_pred_instances = 0
    for instance_id in gt2pred.keys():
        gt2pred[instance_id]['matched_pred'] = []

    # go thru all prediction masks
    for pred_inst_id, pred_inst in pred_instances.items():
        pred_ids = np.array(pred_inst['lines_ids'])
        matched_gt = []  # 当前预测匹配的真实实例
        # 计算与背景的交集
        if len(void_gt_ids) > 0:
            void_intersection = len(np.intersect1d(pred_ids, void_gt_ids))
        else:
            void_intersection = 0
        pred_inst['void_intersection'] = void_intersection        
        # 计算与gt实例的交集
        # go thru all gt instances with matching label
        for gt_inst_id, gt_inst in gt2pred.items():
            gt_ids = np.array(gt_inst['lines_ids'])
            # 计算交集：同时属于预测掩码和真实实例的线数
            intersection = len(np.intersect1d(pred_ids, gt_ids))
            if intersection > 0:  # 有重叠
                gt_copy = gt_inst.copy()
                pred_copy = pred_inst.copy()
                gt_copy['intersection']   = intersection
                pred_copy['intersection'] = intersection
                # 添加到匹配列表
                matched_gt.append(gt_copy)
                # 添加到真实实例的匹配预测列表
                gt2pred[gt_inst_id]['matched_pred'].append(pred_copy)
        pred_inst['matched_gt'] = matched_gt  # 设置匹配的真实实例
        num_pred_instances += 1                   # 预测实例计数增加
        # 添加到预测-> 真实映射
        pred2gt[pred_inst_id] = pred_inst
    return gt2pred, pred2gt

##--------------- Stage2: 精度计算，基于匹配结果计算AP值 ---------------##
def evaluate_matches(match):
    # 评估参数
    overlaps = OVERLAPS  # IoU阈值
    min_region_sizes = [MIN_REGION_SIZES[0]]  # 实例的最小线数
    dist_threshes = [DISTANCE_THRESHES[0]]  # 距离阈值
    dist_confs = [DISTANCE_CONFS[0]]  # 距离置信度

    # 结果矩阵：距离阈值 × IoU阈值
    ap = np.zeros((len(dist_threshes), len(overlaps)), np.float64)
    ##---------------------- 遍历所有评估配置 ----------------------##
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):  # 对每个IoU阈值
            # 标记所有预测实例为未访问
            pred_visited = {}
            for pred_inst_id in match['pred'].keys():
                pred_visited[pred_inst_id] = False
            y_true = np.empty(0)  # 存储标签：1=TP，0=FP
            y_score = np.empty(0)  # 存储对应的置信度
            hard_false_negatives = 0  # 难以检测的假阴性计数
            has_gt = False
            has_pred = False

            pred_instances = match['pred']
            gt_instances = match['gt']

            # 过滤真值实例（大小、距离等条件）
            gt_instances = {gt_id : gt for gt_id, gt in gt_instances.items() if
                            gt['lines_count'] >= min_region_size and gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf}
            if gt_instances:
                has_gt = True
            if pred_instances:
                has_pred = True

            # 初始化当前场景的数组
            cur_true = np.ones(len(gt_instances))  # 初始假设所有真实实例都能匹配
            cur_score = np.ones(len(gt_instances)) * (-float("inf"))  # 初始置信度
            cur_match = np.zeros(len(gt_instances), dtype=np.bool_)  # 匹配标志
            # collect matches：真值实例匹配
            for (gti, gt) in enumerate(gt_instances.values()):
                found_match = False
                num_pred = len(gt['matched_pred'])
                ##--------- 遍历所有可能匹配的预测 ---------##
                for pred in gt['matched_pred']:
                    # greedy assignments
                    if pred_visited[pred['instance_id']]:  # 如果预测已被使用
                        continue  # 跳过，避免重复使用预测
                    
                    # 计算IoU
                    overlap = float(pred['intersection']) / (
                    gt['lines_count'] + pred['lines_count'] - pred['intersection'])
                    
                    if overlap > overlap_th:  # IoU超过阈值
                        if 'confidence' in pred:
                            confidence = pred['confidence']
                        else:
                            confidence = 0.0 # 没有置信度信息时，设为0
                        # if already have a prediction for this gt,
                        # the prediction with the lower score is automatically a false positive
                        if cur_match[gti]:  # 已经有一个匹配结果（重复匹配）
                            # 保留置信度更高的作为TP，较低的作为FP
                            max_score = max(cur_score[gti], confidence)
                            min_score = min(cur_score[gti], confidence)
                            cur_score[gti] = max_score
                            # append false positive
                            cur_true = np.append(cur_true, 0)
                            cur_score = np.append(cur_score, min_score)
                            cur_match = np.append(cur_match, True)
                        # otherwise set score
                        else:  #首次匹配
                            found_match = True
                            cur_match[gti] = True
                            cur_score[gti] = confidence
                            pred_visited[pred['instance_id']] = True  # 标记为已使用
                if not found_match:  # 没有找到匹配的预测
                    hard_false_negatives += 1  # 假阴性计数增加
            # 只保留成功匹配的真实实例
            # remove non-matched ground truth instances
            cur_true = cur_true[cur_match == True]
            cur_score = cur_score[cur_match == True]

            # collect non-matched predictions as false positive
            # 收集未匹配的预测FP
            for pred_id, pred in pred_instances.items():
                found_gt = False
                # 检查是否与任何真实实例匹配
                for gt in pred['matched_gt']:
                    overlap = float(gt['intersection']) / (
                    gt['lines_count'] + pred['lines_count'] - gt['intersection'])
                    if overlap > overlap_th:
                        found_gt = True
                        break
                # 没有匹配的真实实例
                if not found_gt:
                    '''
                    # 计算与无效区域的交集比例
                    num_ignore = pred['void_intersection']
                    for gt in pred['matched_gt']:
                        # group?
                        if gt['instance_id'] < 1000:
                            num_ignore += gt['intersection']
                        # small ground truth instances
                        if gt['lines_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt['dist_conf'] < distance_conf:
                            num_ignore += gt['intersection']
                    proportion_ignore = float(num_ignore) / pred['lines_count']
                    
                    # 如果无效区域比例不超过阈值，视为FP
                    # 忽略与无效区域有过多重叠的预测
                    # if not ignored append false positive
                    if proportion_ignore <= overlap_th:
                    '''
                    proportion_ignore = pred['void_intersection'] / pred['lines_count']
                    # 如果无法忽略，则视为FP
                    if proportion_ignore <= overlap_th:
                        cur_true = np.append(cur_true, 0)  # FP标签
                        if 'confidence' in pred:
                            confidence = pred['confidence']
                        else:
                            confidence = 0.0 # 没有置信度信息时，设为0
                        cur_score = np.append(cur_score, confidence)  # 置信度

            # 添加到总体结果
            # append to overall results
            y_true = np.append(y_true, cur_true)
            y_score = np.append(y_score, cur_score)

            ##-------------------- 精确率-召回率曲线计算 --------------------##
            # compute average precision
            if has_gt and has_pred:
                # compute precision recall curve first

                # sorting and cumsum
                score_arg_sort = np.argsort(y_score)
                y_score_sorted = y_score[score_arg_sort]
                y_true_sorted = y_true[score_arg_sort]
                y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                # unique thresholds
                (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                num_prec_recall = len(unique_indices) + 1

                # prepare precision recall
                num_examples = len(y_score_sorted)
                if(len(y_true_sorted_cumsum) == 0):
                    num_true_examples = 0
                else:
                    num_true_examples = y_true_sorted_cumsum[-1]
                precision = np.zeros(num_prec_recall)
                recall = np.zeros(num_prec_recall)

                # deal with the first point
                y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                # deal with remaining
                for idx_res, idx_scores in enumerate(unique_indices):
                    cumsum = y_true_sorted_cumsum[idx_scores - 1]
                    tp = num_true_examples - cumsum
                    fp = num_examples - idx_scores - tp
                    fn = cumsum + hard_false_negatives
                    p = float(tp) / (tp + fp)
                    r = float(tp) / (tp + fn)
                    precision[idx_res] = p
                    recall[idx_res] = r

                # first point in curve is artificial
                precision[-1] = 1.
                recall[-1] = 0.

                # compute average of precision-recall curve
                recall_for_conv = np.copy(recall)
                recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                recall_for_conv = np.append(recall_for_conv, 0.)

                stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                # integrate is now simply a dot product
                ap_current = np.dot(precision, stepWidths)

            elif has_gt:
                ap_current = 0.0
            else:
                ap_current = float('nan')
            ap[di, oi] = ap_current
    return ap


def visualize_pr_curve(match):
    """
    在 evaluate_matches 的结果基础上绘制精确率-召回率曲线
    """
    overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
    pr_data = {}

    for overlap_th in overlaps:
        y_true = np.empty(0)
        y_score = np.empty(0)
        gt_instances = match['gt']
        pred_instances = match['pred']

        # 收集预测结果
        for gt in gt_instances.values():
            for pred in gt['matched_pred']:
                overlap = float(pred['intersection']) / (
                    gt['lines_count'] + pred['lines_count'] - pred['intersection'])
                if overlap > overlap_th:
                    conf = pred.get('confidence', 0.0)
                    y_true = np.append(y_true, 1)
                    y_score = np.append(y_score, conf)
                else:
                    conf = pred.get('confidence', 0.0)
                    y_true = np.append(y_true, 0)
                    y_score = np.append(y_score, conf)

        # 如果没有匹配数据则跳过
        if len(y_true) == 0:
            continue

        # 计算精确率-召回率曲线
        from sklearn.metrics import precision_recall_curve, auc
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        pr_data[overlap_th] = (precision, recall, pr_auc)

    # 绘制图像
    plt.figure(figsize=(8, 6))
    for overlap_th, (precision, recall, pr_auc) in pr_data.items():
        plt.plot(recall, precision, label=f'IoU={overlap_th:.2f} (AUC={pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_results(avgs, method_name):
    from utils.log import logger
    sep     = ""
    col1    = ":"
    lineLen = 64

    logger.info("")
    logger.info("#" * lineLen)
    logger.info(f"Evaluation Results for Method: {method_name}")
    logger.info("#" * lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    logger.info(line)
    #logger.info("#" * lineLen)

    all_ap_avg  = avgs["all_ap"]
    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    logger.info("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg)  + sep
    line += "{:>15.3f}".format(all_ap_50o)  + sep
    line += "{:>15.3f}".format(all_ap_25o)  + sep
    logger.info(line)
    logger.info("")

##--------------- Stage3: 结果汇总 ---------------##
def compute_averages(aps):
    d_inf = 0
    o50   = np.where(np.isclose(OVERLAPS,0.5))
    o25   = np.where(np.isclose(OVERLAPS,0.25))
    oAllBut25  = np.where(np.logical_not(np.isclose(OVERLAPS,0.25)))
    avg_dict = {}
    #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap']     = np.nanmean(aps[ d_inf,oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,o25])
    return avg_dict


if __name__ == "__main__":
    config = get_config_eval()
    path_manager = PathManager(config['workspace_path'], config['scene_name'])
    if config['graph_clustering'] != '':
        method_name = f"{config['clustering_method']}_{config['graph_clustering']}"
        pred_file = path_manager.get_lines3d_clusters_path(config['clustering_method']+'_'+config['graph_clustering'])
    else:
        method_name = f"{config['clustering_method']}"
        pred_file = path_manager.get_lines3d_clusters_path(config['clustering_method'])
    

    gt_file = '/home/rylynn/Pictures/Clustering_Workspace/Shanghai_Region5/Groundtruth/lines3d_clusters_gt.json'
    line3dpp_path = '/home/rylynn/Pictures/Clustering_Workspace/Shanghai_Region5/Line3D++'
    lines3d, _, _ = parse_lines3dpp(line3dpp_path)

    gt2pred, pred2gt = assign_instances(gt_file, lines3d, pred_file)
    matches ={}
    matches['gt'] = gt2pred
    matches['pred'] = pred2gt

    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    print_results(avgs, method_name)
    #visualize_pr_curve(matches)
