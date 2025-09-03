import torch
import random
from itertools import combinations

def create_neighborhood_subgraph(node_ind, edge_ind, rel_ind,max_nodes,max_relations):
    Batchsize = node_ind.size(0)
    # 定义子图的数量
    num_subgraphs = max_relations*2


    # 存储子图的列表
    subgraphs = []

    # 创建子图
    for _ in range(num_subgraphs):
        # 随机选择初始节点数量，至少1个，最多5个
        num_initial_nodes = random.randint(1, max_nodes-1)
        subgraph_nodes = torch.randint(0, max_nodes-1, (Batchsize, num_initial_nodes))  # 每个批次选若干个初始节点

        # 初始化子图的节点和边
        subgraph_node_indices = []
        subgraph_edge_indices = []
        subgraph_rel_indices = []

        for batch_idx in range(Batchsize):
            # 获取当前批次的初始节点
            initial_nodes = subgraph_nodes[batch_idx].tolist()

            # 去除重复节点
            unique_nodes = list(set(initial_nodes))

            # 记录子图的节点索引
            subgraph_node_indices.append(unique_nodes)

            # 记录子图的边索引和对应的rel_ind关系
            subgraph_edges = []
            subgraph_relations = []
            for edge_idx in range(12):  # 只考虑前12条有效边
                src, tgt = rel_ind[batch_idx, edge_idx]
                if src.item() in unique_nodes and tgt.item() in unique_nodes:
                    subgraph_edges.append(edge_idx)
                    subgraph_relations.append((src.item(), tgt.item()))
            subgraph_edge_indices.append(subgraph_edges)
            subgraph_rel_indices.append(subgraph_relations)

        # 构建子图
        subgraph = {
            'node_indices': subgraph_node_indices,
            'edge_indices': subgraph_edge_indices,
            'rel_indices': subgraph_rel_indices
        }
        subgraphs.append(subgraph)



    # 转换为张量
    node_indices_tensor = torch.full((Batchsize, num_subgraphs, max_nodes), max_nodes-1, dtype=torch.int)
    edge_indices_tensor = torch.full((Batchsize, num_subgraphs, max_relations), max_relations-1, dtype=torch.int)
    rel_indices_tensor = torch.full((Batchsize, num_subgraphs, max_relations, 2), max_nodes-1, dtype=torch.int)

    for subgraph_idx, subgraph in enumerate(subgraphs):
        for batch_idx in range(Batchsize):
            nodes = subgraph['node_indices'][batch_idx]
            edges = subgraph['edge_indices'][batch_idx]
            rels = subgraph['rel_indices'][batch_idx]

            # 填充节点索引
            node_indices_tensor[batch_idx, subgraph_idx, :len(nodes)] = torch.tensor(nodes)

            # 填充边索引
            edge_indices_tensor[batch_idx, subgraph_idx, :len(edges)] = torch.tensor(edges)

            # 填充关系索引
            if rels:  # 检查rels是否为空
                rel_indices_tensor[batch_idx, subgraph_idx, :len(rels)] = torch.tensor(rels)

    # 创建注意力掩码张量
    att_masks = torch.zeros((Batchsize, num_subgraphs, max_nodes), dtype=torch.float)

    for batch_idx in range(Batchsize):
        for subgraph_idx in range(num_subgraphs):
            nodes = node_indices_tensor[batch_idx, subgraph_idx]
            valid_mask = (nodes != (max_nodes-1)).float()  # 有效节点为1，无效节点为0
            att_masks[batch_idx, subgraph_idx] = valid_mask

    # 将注意力掩码转换为对角线形式
    pool_mtx = torch.diag_embed(att_masks)


    return node_indices_tensor, edge_indices_tensor, rel_indices_tensor, att_masks, pool_mtx

def create_all_subgraph(node_ind, edge_ind, rel_ind,max_nodes,max_relations):
    Batchsize = node_ind.size(0)
    num_subgraphs = 0

    # 计算所有可能的子图数量
    for i in range(0, max_nodes):  # 包括选择0个节点的情况
        num_subgraphs += len(list(combinations(range(max_nodes-1), i)))

    # 存储子图的列表
    subgraphs = []

    # 创建子图
    for i in range(0, max_nodes):
        for comb in combinations(range(max_nodes-1), i):
            subgraph_nodes = list(comb)

            # 初始化子图的节点和边
            subgraph_node_indices = []
            subgraph_edge_indices = []
            subgraph_rel_indices = []

            for batch_idx in range(Batchsize):
                # 记录子图的节点索引
                subgraph_node_indices.append(subgraph_nodes)

                # 记录子图的边索引和对应的rel_ind关系
                subgraph_edges = []
                subgraph_relations = []
                for edge_idx in range(max_relations-1):  # 只考虑前12条有效边
                    src, tgt = rel_ind[batch_idx, edge_idx]
                    if src.item() in subgraph_nodes and tgt.item() in subgraph_nodes:
                        subgraph_edges.append(edge_idx)
                        subgraph_relations.append((src.item(), tgt.item()))
                subgraph_edge_indices.append(subgraph_edges)
                subgraph_rel_indices.append(subgraph_relations)

            # 构建子图
            subgraph = {
                'node_indices': subgraph_node_indices,
                'edge_indices': subgraph_edge_indices,
                'rel_indices': subgraph_rel_indices
            }
            subgraphs.append(subgraph)

    # 转换为张量
    node_indices_tensor = torch.full((Batchsize, num_subgraphs, max_nodes), max_nodes-1, dtype=torch.int)
    edge_indices_tensor = torch.full((Batchsize, num_subgraphs, max_relations), max_relations-1, dtype=torch.int)
    rel_indices_tensor = torch.full((Batchsize, num_subgraphs, max_relations, 2), max_nodes-1, dtype=torch.int)

    # 填充张量
    valid_subgraph_count = [0] * Batchsize
    for subgraph_idx, subgraph in enumerate(subgraphs):
        for batch_idx in range(Batchsize):
            nodes = subgraph['node_indices'][batch_idx]
            edges = subgraph['edge_indices'][batch_idx]
            rels = subgraph['rel_indices'][batch_idx]

            # 填充节点索引
            valid_nodes = len(nodes)
            if valid_nodes > 0:
                node_indices_tensor[batch_idx, valid_subgraph_count[batch_idx], :valid_nodes] = torch.tensor(nodes)
                valid_subgraph_count[batch_idx] += 1
            else:
                # 如果子图是空集，确保它放在末尾
                node_indices_tensor[batch_idx, num_subgraphs - 1, :] = (max_nodes-1)

            # 填充边索引
            valid_edges = len(edges)
            if valid_edges > 0:
                edge_indices_tensor[batch_idx, valid_subgraph_count[batch_idx] - 1, :valid_edges] = torch.tensor(edges)

            # 填充关系索引
            if rels:  # 检查rels是否为空
                valid_rels = len(rels)
                if valid_rels > 0:
                    rel_indices_tensor[batch_idx, valid_subgraph_count[batch_idx] - 1, :valid_rels] = torch.tensor(rels)

    # 创建注意力掩码张量
    att_masks = torch.zeros((Batchsize, num_subgraphs, max_nodes), dtype=torch.float)

    for batch_idx in range(Batchsize):
        for subgraph_idx in range(num_subgraphs):
            nodes = node_indices_tensor[batch_idx, subgraph_idx]
            valid_mask = (nodes != (max_nodes-1)).float()  # 有效节点为1，无效节点为0
            att_masks[batch_idx, subgraph_idx] = valid_mask

    # 将注意力掩码转换为对角线形式
    pool_mtx = torch.diag_embed(att_masks)


    return node_indices_tensor, edge_indices_tensor, rel_indices_tensor, att_masks, pool_mtx
