import torch
import torch.nn as nn
import torch.nn.functional as F


class KGATLayer(nn.Module):
    def __init__(self, dim):
        """
        单层 K-GAT 模型
        Args:
            dim (int): 节点和边特征的固定维度
        """
        super(KGATLayer, self).__init__()
        self.node_fc = nn.Linear(dim, dim, bias=False)  # 节点特征的线性映射
        self.edge_fc = nn.Linear(dim, dim, bias=False)  # 边特征的线性映射
        self.attention_fc = nn.Linear(2 * dim + dim + 1, 1, bias=False)  # 注意力权重计算
        self.norm = nn.LayerNorm(dim)  # 归一化层

    def forward(self, obj_nodes, pred_emb, rel_ind, similarity_matrix, nodes_mask, edges_mask):
        """
        执行单层 K-GAT 消息传递和特征更新
        Args:
            obj_nodes (Tensor): 节点特征 (batch_size, num_nodes, dim)
            pred_emb (Tensor): 边特征 (batch_size, num_edges, dim)
            rel_ind (Tensor): 边索引 (batch_size, num_edges, 2)
            similarity_matrix (Tensor): 边的相关性矩阵 (batch_size, num_edges)
            nodes_mask (Tensor): 节点掩码 (batch_size, num_nodes)
            edges_mask (Tensor): 边掩码 (batch_size, num_edges)

        Returns:
            updated_obj_nodes (Tensor): 更新后的节点特征 (batch_size, num_nodes, dim)
        """
        batch_size, num_nodes, dim = obj_nodes.shape
        _, num_edges, _ = pred_emb.shape

        # 映射节点和边特征
        transformed_nodes = self.node_fc(obj_nodes)  # (batch_size, num_nodes, dim)
        transformed_edges = self.edge_fc(pred_emb)  # (batch_size, num_edges, dim)

        updated_obj_nodes = torch.zeros_like(transformed_nodes)

        for b in range(batch_size):
            # 筛选有效边
            valid_rel_ind = rel_ind[b][edges_mask[b]]  # (num_valid_edges, 2)
            valid_edges = transformed_edges[b][edges_mask[b]]  # (num_valid_edges, dim)
            valid_similarity = similarity_matrix[b][edges_mask[b]].unsqueeze(-1)  # (num_valid_edges, 1)

            # 获取源节点和目标节点特征
            src_idx, dst_idx = valid_rel_ind[:, 0], valid_rel_ind[:, 1]
            src_features = transformed_nodes[b][src_idx]  # (num_valid_edges, dim)
            dst_features = transformed_nodes[b][dst_idx]  # (num_valid_edges, dim)

            # 拼接特征
            attention_input = torch.cat([src_features, dst_features, valid_similarity, valid_edges], dim=-1)
            assert attention_input.shape[-1] == 2 * dim + dim + 1, "Attention Input Dimension Mismatch!"

            # 计算注意力权重
            attention_scores = F.leaky_relu(self.attention_fc(attention_input))  # (num_valid_edges, 1)
            attention_weights = F.softmax(attention_scores, dim=0)  # (num_valid_edges, 1)

            # 消息传递
            messages = attention_weights * src_features  # (num_valid_edges, dim)
            for i in range(valid_rel_ind.size(0)):
                updated_obj_nodes[b, dst_idx[i]] += messages[i]

        # 屏蔽无效节点特征
        updated_obj_nodes = updated_obj_nodes * nodes_mask.unsqueeze(-1)
        return self.norm(updated_obj_nodes)


class MultiLayerKGAT(nn.Module):
    def __init__(self, dim, num_layers):
        """
        多层 K-GAT 模型
        Args:
            dim (int): 节点和边特征的固定维度
            num_layers (int): 堆叠的 K-GAT 层数
        """
        super(MultiLayerKGAT, self).__init__()
        self.kgat_layers = nn.ModuleList([KGATLayer(dim) for _ in range(num_layers)])  # 所有层使用相同维度

    def forward(self, obj_nodes, pred_emb, rel_ind, similarity_matrix, nodes_mask, edges_mask):
        """
        执行多层 K-GAT
        """
        x = obj_nodes
        for i, kgat_layer in enumerate(self.kgat_layers):
            x = kgat_layer(x, pred_emb, rel_ind, similarity_matrix, nodes_mask, edges_mask)
        return x
