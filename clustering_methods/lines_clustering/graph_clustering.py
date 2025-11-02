import igraph as ig
import random
import networkx as nx
from node2vec import Node2Vec
import networkx as nx
import numpy as np
import hdbscan
from scipy.sparse import csr_matrix

def leiden_community(edges, weights, num_segments, idx_to_segment):

    # 创建图
    g = ig.Graph()
    g.add_vertices(num_segments)
    g.add_edges(edges)
    g.es['weight'] = weights

    # 使用Leiden算法进行社区发现
    partition = g.community_leiden(
        objective_function='modularity',
        weights='weight',
        resolution_parameter=1.0,
        n_iterations=-1  # 直到收敛
    )
    
    # 收集聚类结果
    clusters = {}
    for idx, label in enumerate(partition.membership):
        seg = idx_to_segment[idx]
        clusters.setdefault(label, set()).add(seg)
    # 剔除小于3个线段的聚类
    clusters = {k: list(v) for k, v in clusters.items() if len(v) >= 3}
    return clusters


def chinese_whispers(edges, weights, num_segments, idx_to_segment, iterations=20):
    """
    使用 Chinese Whispers 算法（无监督图聚类）
    """
    G = nx.Graph()
    for (u, v), w in zip(edges, weights):
        G.add_edge(u, v, weight=w)

    # 初始化每个节点的标签为自身
    labels = {n: n for n in range(num_segments)}

    for it in range(iterations):
        nodes = list(G.nodes())
        random.shuffle(nodes)
        for n in nodes:
            # 根据邻居的加权投票更新标签
            neigh_labels = {}
            for nb in G.neighbors(n):
                lbl = labels[nb]
                w = G[n][nb]["weight"]
                neigh_labels[lbl] = neigh_labels.get(lbl, 0) + w
            if neigh_labels:
                # 选得票最高的标签
                labels[n] = max(neigh_labels.items(), key=lambda x: x[1])[0]
        # 可加收敛检测

    # 压缩为连续标签
    label_map = {l: i for i, l in enumerate(set(labels.values()))}
    # 形成线段聚类结果
    clusters = {}
    for n, l in labels.items():
        clabel = label_map[l]
        seg = idx_to_segment[n]
        clusters.setdefault(clabel, set()).add(seg)
    # 剔除小于3个线段的聚类
    clusters = {k: list(v) for k, v in clusters.items() if len(v) >= 3}
    return clusters


def node2vec_hdbscan(edges, weights, num_segments, idx_to_segment,
                              dimensions=64, walk_length=20, num_walks=10,
                              min_cluster_size=10):
    """
    Node2Vec 嵌入 + HDBSCAN 聚类
    """
    G = nx.Graph()
    for (u, v), w in zip(edges, weights):
        G.add_edge(u, v, weight=w)

    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, workers=4, weight_key="weight")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    emb = np.array([model.wv[str(i)] if str(i) in model.wv else np.zeros(dimensions) for i in range(num_segments)])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(emb)
    # 形成线段聚类结果
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # 噪声点
        seg = idx_to_segment[idx]
        clusters.setdefault(int(label), set()).add(seg)
    # 剔除小于3个线段的聚类
    clusters = {k: list(v) for k, v in clusters.items() if len(v) >= 3}
    return clusters


def direct_hdbscan(edges, weights, num_segments, idx_to_segment, min_cluster_size=10):
    """
    直接将边权视为相似度进行 HDBSCAN 聚类
    """
    # 稀疏邻接矩阵
    row, col = zip(*edges)
    adj = csr_matrix((weights, (row, col)), shape=(num_segments, num_segments))
    adj = adj + adj.T  # 确保对称
    features = adj.toarray()  # 注意：若节点很多会占内存（3万节点要谨慎）

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(features)
    # 形成线段聚类结果
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # 噪声点
        seg = idx_to_segment[idx]
        clusters.setdefault(int(label), set()).add(seg)
    # 剔除小于3个线段的聚类
    clusters = {k: list(v) for k, v in clusters.items() if len(v) >= 3}
    return clusters