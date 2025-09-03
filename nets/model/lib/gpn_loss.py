from nets.encoder.clip import clip
import torch
import pandas as pd
from nltk.corpus import wordnet as wn
# 生成 Ground-truth 特征
def get_ground_truth_features(gt_nodes_name, model, device):
    ground_truth_features = []
    for nouns in gt_nodes_name:
        prompts = [f"a {noun}" for noun in nouns]
        tokens = clip.tokenize(prompts).to(device)
        if tokens.shape[0] != 1:
            tokens = tokens.squeeze()
        with torch.no_grad():
            text_features = model.encode_text(tokens).to(device)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        ground_truth_features.append(text_features)
    return ground_truth_features


# 计算相似度并标记有效节点
import torch

def calculate_similarity_and_mark_effective_nodes(obj_nodes, ground_truth_features, valid_threshold):
    """
    此函数用于计算相似度并标记有效的节点
    :param obj_nodes: 对象节点特征张量，形状为 (batch_size, num_nodes, feature_dim)
    :param ground_truth_features: 基准特征张量，形状为 (batch_size, num_gt, feature_dim)
    :param valid_threshold: 有效节点的相似度阈值
    :return: 有效节点的索引列表，形状为 (batch_size, num_effective_nodes)
    """
    # 检查并调整 obj_nodes 的维度
    if len(obj_nodes.shape) == 2:
        obj_nodes = obj_nodes.unsqueeze(0)  # 添加batch维度
    batch_size = obj_nodes.shape[0]
    effective_nodes = []

    """
    """
    for batch_idx in range(batch_size):
        node_features = obj_nodes[batch_idx]
        gt_features = ground_truth_features[batch_idx]
        num_nodes = node_features.shape[0]
        batch_effective_nodes = []

        if gt_features.shape[0] == 1:
            # 特殊处理，当 gt_features 只有一个元素时
            for node_idx in range(num_nodes):
                node_feature = node_features[node_idx]
                similarity = (100.0 * torch.dot(node_feature, gt_features.squeeze())).item()
                if similarity > valid_threshold:
                    batch_effective_nodes.append(node_idx)
        else:
            for node_idx in range(num_nodes):
                node_feature = node_features[node_idx].unsqueeze(0)
                # 计算当前节点特征与所有 Ground-truth 特征的余弦相似度
                similarity = (100.0 * torch.matmul(node_feature, gt_features.T)).softmax(dim=-1)
                # 查找相似度最高的 Ground-truth 特征
                max_similarity, _ = torch.max(similarity, dim=-1)

                if max_similarity > valid_threshold:
                    batch_effective_nodes.append(node_idx)

        effective_nodes.append(batch_effective_nodes)


    return effective_nodes

# 计算子图的交并比并标记子图
def calculate_iou_and_mark_subgraphs(obj_ind, effective_nodes, iou_threshold,nodes_range):
    batch_size = obj_ind.shape[0]
    num_subgraphs = obj_ind.shape[1]
    subgraph_labels = []

    for batch_idx in range(batch_size):
        subgraph_nodes = obj_ind[batch_idx]
        batch_effective_nodes = set(effective_nodes[batch_idx])

        for subgraph_idx in range(num_subgraphs):
            valid_indices = subgraph_nodes[subgraph_idx][subgraph_nodes[subgraph_idx] != nodes_range]
            valid_indices_set = set(valid_indices.tolist())

            intersection = len(batch_effective_nodes & valid_indices_set)
            union = len(batch_effective_nodes | valid_indices_set)

            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union

            subgraph_labels.append((batch_idx, subgraph_idx, iou > iou_threshold))

    return subgraph_labels



def get_valid_nodes_fromWN(nodes_name, gt_nodes_name, nodes_range,threshold):
    """
    根据名词相似度计算，标记有效节点的索引。

    Parameters:
        nodes_name (list of list of str): 输入的节点名列表，每个子列表包含名词。
        gt_nodes_name (list of list of str): 对比的目标节点名列表，每个子列表包含名词。
        threshold (float): 名词相似度的阈值，默认0.85。

    Returns:
        list of list of int: 每个子列表中有效节点的索引。
    """
    import nltk
    """
    import os
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    nltk.download('wordnet')
    """
    # 1. 截取每个子列表的前nodes_range个有效名词,这里为6

    new_nodes_name = [lst[:nodes_range] for lst in nodes_name]

    # 计算两个名词的 WordNet 相似度
    def compute_similarity(word1, word2):
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        if not synsets1 or not synsets2:
            return 0  # 如果任一词语没有同义词集，则相似性为 0
        max_sim = max((s1.wup_similarity(s2) or 0) for s1 in synsets1 for s2 in synsets2)
        return max_sim

    # 2. 比较新的 nodes_name 和 gt_nodes_name 的名词，判断有效节点
    result = []

    for i, node_list in enumerate(new_nodes_name):
        valid_indices = []
        for j, word in enumerate(node_list):
            for gt_word in gt_nodes_name[i]:
                sim = compute_similarity(word, gt_word)
                if sim and sim > threshold:
                    valid_indices.append(j)
                    break  # 如果已经标记为有效，就无需再比较下一个 gt_word
        result.append(valid_indices)

    return result
def prompt2nodes(clip_model,gt_nodes_name, obj_ind, obj_nodes,device,valid_threshold,iou_threshold,using_WN,nodes_name,nodes_range):

    if not using_WN:
        ground_truth_features = get_ground_truth_features(gt_nodes_name, clip_model, device)
        effective_nodes = calculate_similarity_and_mark_effective_nodes(obj_nodes, ground_truth_features, valid_threshold)

    else:
        effective_nodes = get_valid_nodes_fromWN(nodes_name, gt_nodes_name, nodes_range,valid_threshold)

    subgraph_labels = calculate_iou_and_mark_subgraphs(obj_ind, effective_nodes, iou_threshold,nodes_range)



    # 提取 batch_idx, subgraph_idx 和 is_positive
    batch_indices = [label[0] for label in subgraph_labels]
    subgraph_indices = [label[1] for label in subgraph_labels]
    is_positive = [label[2] for label in subgraph_labels]

    # 将列表转换为张量
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    subgraph_indices = torch.tensor(subgraph_indices, dtype=torch.long)
    is_positive = torch.tensor(is_positive, dtype=torch.bool)

    # 初始化一个形状为 (2, 26) 的全零矩阵
    score_target = torch.zeros((obj_ind.size(0), obj_ind.size(1)), dtype=torch.int)

    # 使用索引和布尔掩码填充 feature_matrix
    positive_indices = is_positive.nonzero().squeeze()
    if positive_indices.numel() > 0:
        score_target[batch_indices[positive_indices], subgraph_indices[positive_indices]] = 1
    # dummy sub
    score_target[:,-1] = 0

    # 打印结果
    return score_target