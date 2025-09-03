from nets.encoder.clip import clip
import torch
import pandas as pd
# 生成 Ground-truth 特征
def get_ground_truth_features(noun_lists, model, device):
    ground_truth_features = []
    for nouns in noun_lists:
        prompts = [f"a {noun}" for noun in nouns]
        tokens = clip.tokenize(prompts).to(device)
        if tokens.shape[0] != 1:
            tokens = tokens.squeeze()
        with torch.no_grad():
            text_features = model.encode_text(tokens).to(device)
        ground_truth_features.append(text_features)
    return ground_truth_features


# 计算相似度并标记有效节点
def calculate_similarity_and_mark_effective_nodes(obj_nodes, ground_truth_features, valid_threshold):
    batch_size = obj_nodes.shape[0]
    effective_nodes = []

    for batch_idx in range(batch_size):
        node_features = obj_nodes[batch_idx]
        gt_features = ground_truth_features[batch_idx]

        # 计算每个节点与所有 Ground-truth 特征的余弦相似度
        similarities = torch.cosine_similarity(node_features.unsqueeze(1), gt_features, dim=-1)

        # 查找相似度最高的 Ground-truth 特征
        max_similarities, _ = torch.max(similarities, dim=1)

        # 标记相似度高于阈值的节点
        effective_mask = max_similarities > valid_threshold
        effective_indices = torch.nonzero(effective_mask).squeeze()

        # 确保 effective_indices 是一个列表
        if effective_indices.numel() == 0:
            effective_indices = []
        elif effective_indices.numel() == 1:
            effective_indices = [effective_indices.item()]
        else:
            effective_indices = effective_indices.tolist()



        effective_nodes.append(effective_indices)

    return effective_nodes

# 计算子图的交并比并标记子图
def calculate_iou_and_mark_subgraphs(obj_ind, effective_nodes, iou_threshold):
    batch_size = obj_ind.shape[0]
    num_subgraphs = obj_ind.shape[1]
    subgraph_labels = []

    for batch_idx in range(batch_size):
        subgraph_nodes = obj_ind[batch_idx]
        batch_effective_nodes = set(effective_nodes[batch_idx])

        for subgraph_idx in range(num_subgraphs):
            valid_indices = subgraph_nodes[subgraph_idx][subgraph_nodes[subgraph_idx] != 6]
            valid_indices_set = set(valid_indices.tolist())

            intersection = len(batch_effective_nodes & valid_indices_set)
            union = len(batch_effective_nodes | valid_indices_set)

            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union

            subgraph_labels.append((batch_idx, subgraph_idx, iou > iou_threshold))

    return subgraph_labels

def prompt2nodes(clip_model,prompt_list, obj_ind, obj_nodes,device,valid_threshold,iou_threshold):
    xlsx_file = '/home/ippl/LFY/TEXT2IMAGE/nets/model/lib/updated_prompts.xlsx'
    df = pd.read_excel(xlsx_file, header=None)

    # 创建一个空字典来存储结果
    prompt_dict = {}

    # 遍历DataFrame，提取prompt和名词列表
    for index, row in df.iterrows():
        prompt = row[2].strip()  # 第三列的prompt
        nouns = row[3]  # 第四列的名词列表
        if isinstance(nouns, str):  # 确保名词列表是字符串
            nouns_list = [noun.strip() for noun in nouns.split(',')]  # 去除名词之间的空格
            prompt_dict[prompt] = nouns_list

    prompts = prompt_dict

    # 存储每个 prompt 对应的所有名词的列表
    noun_lists = []

    # 遍历每个 prompt，提取名词并存储
    for prompt in prompt_list:
        nouns = prompts.get(prompt, [])
        noun_lists.append(nouns)


    ground_truth_features = get_ground_truth_features(noun_lists, clip_model, device)
    effective_nodes = calculate_similarity_and_mark_effective_nodes(obj_nodes, ground_truth_features, valid_threshold)
    subgraph_labels = calculate_iou_and_mark_subgraphs(obj_ind, effective_nodes, iou_threshold)

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