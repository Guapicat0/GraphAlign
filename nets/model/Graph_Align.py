import torch
import torch.nn as nn
from pathlib import Path
import json
from nets.encoder.clip import clip
import ast
from torchvision.transforms import functional as F

import torch.nn as nn
import nets.model.lib.gcn_backbone as GBackbone
import nets.model.lib.gpn as GPN
import random
from nets.model.lib.subgraph_sample import *
from nets.model.lib.SGG_FUNC import split_prompt,process_scene_graphs
from nets.model.lib.gpn_loss import prompt2nodes
from nets.model.lib.get_scenegraph import FeatureExtractor
from nets.model.lib.load_GT_graph import get_matching_graphs_from_prompts
from nets.model.lib.KGAT import MultiLayerKGAT
from nets.model.fuse_modules import BiAttentionBlock

class PostProcess(nn.Module):
    def __init__(self, device, model, preprocess,max_nodes,max_relations,gt_max_nodes,gt_max_relations,graph_node_dim,graph_edge_dim,using_semantic_graph):
        super(PostProcess, self).__init__()
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.dummy_node = 1
        self.max_nodes = max_nodes
        self.max_relations = max_relations
        self.gt_max_nodes = gt_max_nodes
        self.gt_max_relations = gt_max_relations
        self.graph_node_dim = graph_node_dim
        self.graph_edge_dim = graph_edge_dim

        self.get_params = FeatureExtractor(self.device, self.model,self.preprocess,
                                           self.max_nodes, self.max_relations,
                                           self.graph_node_dim,self.graph_edge_dim)
        self.get_gt_params = FeatureExtractor(self.device, self.model,self.preprocess,
                                              self.gt_max_nodes, self.gt_max_relations,
                                              self.graph_node_dim,self.graph_edge_dim)
        self.using_semantic_graph = using_semantic_graph

    def dynamic_neighbor_expansion(self,nodes_name,image_nodes,obj_nodes, pred_emb, rel_ind, nodes_mask, edges_mask, semantic_graph):
        """
        扩展动态邻居，根据语义图信息增加语义节点和边。

        Args:
            nodes_name (List): 节点名称列表，每个元素是场景的节点名称列表
            obj_nodes (Tensor): 节点文本嵌入 (batch_size, num_nodes, feature_dim)
            pred_emb (Tensor): 边的文本嵌入 (batch_size, num_edges, feature_dim)
            rel_ind (Tensor): 边的指向关系 (batch_size, num_edges, 2)
            nodes_mask (Tensor): 有效节点掩码 (batch_size, num_nodes)
            edges_mask (Tensor): 有效边掩码 (batch_size, num_edges)
            semantic_graph (List): 语义图信息，包含每个场景图的动态邻居扩展信息

        Returns:
            extend_nodes_name (List): 扩展后的节点名称 (batch_size, new_num_nodes)
            extend_obj_nodes (Tensor): 拓展语义图后节点文本嵌入 (batch_size, new_num_nodes, feature_dim)
            extend_pred_emb (Tensor): 边的文本嵌入 (batch_size, new_num_edges, feature_dim)
            extend_rel_ind (Tensor): 边的指向关系 (batch_size, new_num_edges, 2)
            extend_nodes_mask (Tensor): 有效节点掩码 (batch_size, new_num_nodes)
            extend_edges_mask (Tensor): 有效边掩码 (batch_size, new_num_edges)
            similarity_matrix:边的相关性矩阵(batch_size,new_num_edges)，场景图（原始图）边的相关性为1，
            新增语义图的相关性为 info['similarity']，无效边的相关性为0
        """

        batch_size, num_nodes, feature_dim = obj_nodes.shape
        _, num_edges, edge_feature_dim = pred_emb.shape

        extend_nodes_name = []  # 新增: 用于存储扩展后的节点名称

        extend_obj_nodes = []
        extend_pred_emb = []
        extend_rel_ind = []
        extend_nodes_mask = []
        extend_edges_mask = []
        similarity_matrix = []

        max_num_nodes = 0
        max_num_edges = 0

        # Step 1: 遍历每个 Batch，处理扩展并统计最大节点和边的数量
        for b in range(batch_size):
            # 提取有效节点和边
            valid_nodes = obj_nodes[b][nodes_mask[b]]  # (num_valid_nodes, feature_dim)
            valid_rel_ind = rel_ind[b][edges_mask[b]]  # (num_valid_edges, 2)
            valid_edges = pred_emb[b][edges_mask[b]]  # (num_valid_edges, feature_dim)

            # 初始化扩展后的图
            new_nodes = valid_nodes.clone()
            new_edges = valid_edges.clone()
            new_rel_inds = valid_rel_ind.clone()
            new_similarity = torch.ones(valid_rel_ind.size(0), 1).to(obj_nodes.device)  # 原始边的相关性为1
            new_node_names = list(nodes_name[b])  # 获取当前batch的节点名称
            new_nodes_list = new_node_names.copy()  # 用于存储扩展后的节点名称列表
            # 动态扩展语义邻居
            semantic_info = semantic_graph[b]
            if semantic_info:  # 如果语义图非空
                for info in semantic_info:
                    node_idx = info['node_index']  # 场景图的主语节点索引
                    relation_name = info['relation_name']  # 语义关系名称
                    similarity = info['similarity']  # 相似度
                    new_node_name = info['gt_name']  # 新增的语义节点名称

                    # 对新增的节点文本进行编码
                    node_token = clip.tokenize(new_node_name).to(self.device)
                    if node_token.shape[0] != 1:
                        node_token = node_token.squeeze()
                    with torch.no_grad():
                        text_features = self.model.encode_text(node_token).to(self.device)
                        node_text_features = text_features / text_features.norm(dim=1, keepdim=True)

                    # 对新增的边文本进行编码
                    relation_token = clip.tokenize(relation_name).to(self.device)
                    if relation_token.shape[0] != 1:
                        relation_token = relation_token.squeeze()
                    with torch.no_grad():
                        relation_token_text_features = self.model.encode_text(relation_token).to(self.device)
                        relation_token_text_features = relation_token_text_features / relation_token_text_features.norm(
                            dim=1, keepdim=True)

                    # 添加新增节点
                    new_nodes = torch.cat([new_nodes, node_text_features], dim=0)
                    new_nodes_list.append(new_node_name)
                    new_node_idx = new_nodes.size(0) - 1  # 新节点的索引

                    # 添加新增边
                    new_edges = torch.cat([new_edges, relation_token_text_features], dim=0)
                    new_rel_inds = torch.cat(
                        [new_rel_inds, torch.tensor([[new_node_idx, node_idx]]).to(obj_nodes.device)], dim=0)

                    # 添加新增边的相似度
                    new_similarity = torch.cat([new_similarity, torch.tensor([[similarity]]).to(obj_nodes.device)],
                                               dim=0)

            # 更新最大节点数和边数
            max_num_nodes = max(max_num_nodes, new_nodes.size(0))
            max_num_edges = max(max_num_edges, new_edges.size(0))

            # 保存每个 Batch 的扩展结果
            extend_obj_nodes.append(new_nodes)
            extend_pred_emb.append(new_edges)
            extend_rel_ind.append(new_rel_inds)
            extend_nodes_mask.append(torch.ones(new_nodes.size(0), dtype=torch.bool).to(obj_nodes.device))
            extend_edges_mask.append(torch.ones(new_edges.size(0), dtype=torch.bool).to(obj_nodes.device))
            similarity_matrix.append(new_similarity.squeeze(-1))  # 去掉最后一维
            extend_nodes_name.append(new_nodes_list)  # 保存当前batch的节点名称

        # Step 2: 补充无效节点和边，对齐到最大节点数和最大边数
        for b in range(batch_size):
            # 补充无效节点
            pad_num_nodes = max_num_nodes - extend_obj_nodes[b].size(0)
            if pad_num_nodes > 0:
                pad_nodes = torch.zeros(pad_num_nodes, feature_dim).to(obj_nodes.device)
                pad_node_mask = torch.zeros(pad_num_nodes, dtype=torch.bool).to(obj_nodes.device)

                extend_obj_nodes[b] = torch.cat([extend_obj_nodes[b], pad_nodes], dim=0)
                extend_nodes_mask[b] = torch.cat([extend_nodes_mask[b], pad_node_mask], dim=0)

            # 补充无效边
            pad_num_edges = max_num_edges - extend_pred_emb[b].size(0)
            if pad_num_edges > 0:
                pad_edges = torch.zeros(pad_num_edges, edge_feature_dim).to(obj_nodes.device)
                pad_rel_inds = torch.zeros(pad_num_edges, 2).long().to(obj_nodes.device) + (max_num_nodes - 1)
                pad_edge_mask = torch.zeros(pad_num_edges, dtype=torch.bool).to(obj_nodes.device)
                pad_similarity = torch.zeros(pad_num_edges).to(obj_nodes.device)

                extend_pred_emb[b] = torch.cat([extend_pred_emb[b], pad_edges], dim=0)
                extend_rel_ind[b] = torch.cat([extend_rel_ind[b], pad_rel_inds], dim=0)
                extend_edges_mask[b] = torch.cat([extend_edges_mask[b], pad_edge_mask], dim=0)
                similarity_matrix[b] = torch.cat([similarity_matrix[b], pad_similarity], dim=0)

        # Step 3: 将列表转换为张量
        extend_obj_nodes = torch.stack(extend_obj_nodes, dim=0)
        extend_pred_emb = torch.stack(extend_pred_emb, dim=0)
        extend_rel_ind = torch.stack(extend_rel_ind, dim=0)
        extend_nodes_mask = torch.stack(extend_nodes_mask, dim=0)
        extend_edges_mask = torch.stack(extend_edges_mask, dim=0)
        similarity_matrix = torch.stack(similarity_matrix, dim=0)

        # 补充视觉特征：
        b, x, dim = image_nodes.shape
        _, y, _ = extend_obj_nodes.shape

        if x < y:
            # 计算需要填充的长度
            pad_length = y - x
            # 创建一个零张量用于填充
            pad_tensor = torch.zeros(b, pad_length, dim, device=image_nodes.device)
            # 使用 torch.cat 进行拼接
            extend_image_nodes = torch.cat([image_nodes, pad_tensor], dim=1)
        else:
            extend_image_nodes = image_nodes


        return (
            extend_nodes_name,
            extend_image_nodes,
            extend_obj_nodes,
            extend_pred_emb,
            extend_rel_ind,
            extend_nodes_mask,
            extend_edges_mask,
            similarity_matrix,
        )

    def forward(self, image, prompt,scene_graph,gt_graphs,semantic_graph):

        #  get features from full graph include node,edge features and relation_indices


        gt_nodes = []
        gt_relations = []
        for gt_graph in gt_graphs:
            gt_node = gt_graph['nodes']
            gt_relation = gt_graph['relations']
            gt_nodes.append(gt_node)
            gt_relations.append(gt_relation)

        gt_nodes_name = [[node['name'] for node in batch] for batch in gt_nodes]
        gt_obj_nodes = self.get_gt_params.extract_node_textual_features(gt_nodes)
        gt_pred_emb = self.get_gt_params.extract_edge_textual_features(gt_relations)
        gt_rel_ind = self.get_gt_params.extract_relation_indices(gt_relations)
        gt_nodes_mask,gt_edges_mask = self.get_gt_params.generate_masks(gt_nodes,gt_relations)


        nodes,relations = self.get_params.process_scene_graph(scene_graph)
        nodes_name = [[node['name'] for node in batch] for batch in nodes]
        images_nodes = self.get_params.extract_node_vision_features(image,nodes)
        obj_nodes = self.get_params.extract_node_textual_features(nodes)
        pred_emb = self.get_params.extract_edge_textual_features(relations)
        rel_ind = self.get_params.extract_relation_indices(relations)
        nodes_mask,edges_mask= self.get_params.generate_masks(nodes,relations)

        if self.using_semantic_graph:
            (extend_nodes_name,extend_image_nodes,extend_obj_nodes,extend_pred_emb,extend_rel_ind,extend_nodes_mask,extend_edges_mask,similarity_matrix)\
                = self.dynamic_neighbor_expansion(nodes_name,images_nodes,obj_nodes, pred_emb, rel_ind, nodes_mask, edges_mask, semantic_graph)


        Batchsize = rel_ind.size(0)
        node_ind = torch.randn(Batchsize, self.max_nodes)
        edge_ind = torch.randn(Batchsize, self.max_relations)
        sub_node_ind,sub_edge_ind,sub_rel_ind,att_masks,pool_mtx = create_all_subgraph(node_ind ,edge_ind,rel_ind,
                                                                                       self.max_nodes,self.max_relations)
        # 将注意力掩码转换为对角线形式
        pool_mtx = pool_mtx.to(self.device)
        att_masks = att_masks.to(self.device)

        token = clip.tokenize(prompt).to(self.device)
        if token.shape[0] != 1:
            token = token.squeeze()
        with torch.no_grad():
            text_features = self.model.encode_text(token)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)


        sub_graph = {
            "sub_node_ind": sub_node_ind,
            "sub_edge_ind": sub_edge_ind,
            "sub_rel_ind": sub_rel_ind,
            "att_masks": att_masks,
            "pool_mtx": pool_mtx,
        }

        gt_graph = {
            "prompt": prompt,
            "text_features": text_features,
            "gt_nodes_name": gt_nodes_name,
            "gt_obj_nodes": gt_obj_nodes,
            "gt_pred_emb": gt_pred_emb,
            "gt_rel_ind": gt_rel_ind,
            "gt_nodes_mask": gt_nodes_mask,
            "gt_edges_mask": gt_edges_mask,
        }

        if self.using_semantic_graph:
            pred_graph = {
                "nodes_name": extend_nodes_name,
                "image_nodes": extend_image_nodes,
                "obj_nodes": extend_obj_nodes,
                "pred_emb": extend_pred_emb,
                "rel_ind": extend_rel_ind,
                "nodes_mask": extend_nodes_mask,
                "edges_mask": extend_edges_mask,
                "similarity_matrix": similarity_matrix
            }
        else :
            pred_graph = {
                "nodes_name": nodes_name,
                "image_nodes": images_nodes,
                "obj_nodes": obj_nodes,
                "pred_emb": pred_emb,
                "rel_ind": rel_ind,
                "nodes_mask": nodes_mask,
                "edges_mask": edges_mask,
            }


        return pred_graph,sub_graph,gt_graph



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size * 2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        # 调用权重初始化函数对模型参数进行初始化
        #self.initialize_weights()

    def forward(self, x1, x2):
        # Flatten the input tensors
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate the flattened input tensors
        inputs = torch.cat((x1, x2), dim=1)

        # Pass the inputs through the MLP layers
        hidden = self.fc1(inputs.to(self.fc1.weight.dtype))
        hidden = torch.relu(hidden)
        output = self.fc2(hidden)

        return output

class FFNLayer(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(FFNLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()  # 实例化激活函数
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class OVSGTR_ALIGN(nn.Module):
    def __init__(self,device):
        super(OVSGTR_ALIGN, self).__init__()
        self.device = device
        self.max_nodes = 7    # = max_nodes + dummy_nodes = 6 + 1
        self.max_relations = 13 # max_relations + dummy_relations = 12 + 1
        self.gt_max_nodes =15
        self.gt_max_relations = 30
        self.graph_node_dim = 512
        self.graph_edge_dim = 512

        self.att_feat_size=512
        self.GCN_dim=1024
        self.embed_dim = 512
        self.obj_v_proj = nn.Linear(self.att_feat_size, self.GCN_dim).to(self.device)
        self.obj_emb_proj = nn.Linear(self.embed_dim, self.GCN_dim).to(self.device)
        self.pred_emb_prj = nn.Linear(self.embed_dim, self.GCN_dim).to(self.device)
        self.GCN_layers = 2
        self.GCN_residual = 2
        self.gcn_bn = 0
        self.GCN_use_bn = False if self.gcn_bn == 0 else True
        self.att_hid_size = 512
        self.proj_dim = 2048
        self.valid_threshold = 0.5
        self.iou_threshold = 0.8
        self.dummy_node =1

        script_dir = Path(__file__).parent
        # 向上移动两级目录,到达 'root/model_data/AesCLIP'
        ckpt = str(script_dir.parent.parent/'model_data'/'ViT-B-32.pt')
        self.model, self.preprocess = clip.load(ckpt, device=device)
        self.model = self.model.float()
        self.using_semantic_graph = False
        self.using_gt_graph_encoder = True
        self.using_WN = True
        self.PostProcess = PostProcess(device=self.device, model=self.model, preprocess=self.preprocess,
                                       max_nodes=self.max_nodes, max_relations=self.max_relations,
                                       gt_max_nodes = self.gt_max_nodes, gt_max_relations =self.gt_max_relations,
                                       graph_node_dim = self.graph_node_dim,graph_edge_dim=self.graph_edge_dim,
                                       using_semantic_graph=self.using_semantic_graph)
        # GCN backbone
        self.gcn_backbone = GBackbone.gcn_backbone(GCN_layers=self.GCN_layers, GCN_dim=self.GCN_dim, \
                                                   GCN_residual=self.GCN_residual, GCN_use_bn=self.GCN_use_bn).to(self.device)
        self.GT_gcn_backbone =  GBackbone.gcn_backbone(GCN_layers=self.GCN_layers, GCN_dim=self.GCN_dim, \
                                                   GCN_residual=self.GCN_residual, GCN_use_bn=self.GCN_use_bn).to(self.device)
        self.relu = nn.ReLU(inplace=True)
        self.KAGT = MultiLayerKGAT(dim=1024, num_layers=3).to(self.device)
        self.gpn_layer = GPN.gpn_layer(GCN_dim=self.GCN_dim, hid_dim=self.att_hid_size).to(self.device)

        self.txt_proj= nn.Linear(self.embed_dim, self.proj_dim).to(self.device)
        self.mlp = MLP(self.proj_dim, int(self.proj_dim / 2), 1).to(self.device)
        self.GPN_loss = nn.BCELoss()

        self.GT_obj_emb_proj = nn.Linear(self.embed_dim, self.GCN_dim).to(self.device)
        self.GT_pred_emb_prj = nn.Linear(self.embed_dim, self.GCN_dim).to(self.device)
        self.hid_dim=512
        self.gt_read_out_proj = nn.Sequential(nn.Linear(self.GCN_dim*2, self.hid_dim),
                                           nn.Linear(self.hid_dim, self.GCN_dim*2)).to(self.device)





    def forward(self, image, prompt,scene_graph,gt_graph,semantic_graph):
        image = image.to(self.device)
        pred_graph , sub_graph , gt_graph = self.PostProcess(image,prompt,scene_graph,gt_graph,semantic_graph)

        batchsize = pred_graph["obj_nodes"].size(0)
        # num_features以及num_relations为场景图所包含的有效节点数和边数+ 1 无效节点/边数
        num_features = self.max_nodes
        num_relations = self.max_relations
        L = self.GCN_dim
        vis_emb = self.obj_v_proj(pred_graph["image_nodes"])
        #vis_emb = self.relu(vis_emb)
        # 把512维的节点文本特征（2,7,512)->（2,7,1024）
        obj_emb = self.obj_emb_proj(pred_graph["obj_nodes"])
        obj_emb = self.relu(vis_emb + obj_emb)
        # 把512的边文本特征（2,13,512)->（2,13,1024）
        pred_fmap = self.pred_emb_prj(pred_graph["pred_emb"])

        obj_dist = None
        # STEP-1 : 使用GCN / K-GAT 对 预测出的场景图进行Graph-Encoding
        if not self.using_semantic_graph:
            att_feats, x_pred = self.gcn_backbone(batchsize,num_features,num_relations,L,obj_emb, obj_dist, pred_fmap, pred_graph["rel_ind"])
        else:
            att_feats = self.KAGT(obj_emb, pred_fmap, pred_graph["rel_ind"], pred_graph["similarity_matrix"], pred_graph["nodes_mask"], pred_graph["edges_mask"])
            x_pred = None
            att_feats = att_feats[:,:self.max_nodes,:]


        # STEP-2 : 从场景图中挑选合适的子图
        fc_feats = None

        sub_max_ind, all_subgraph_obj_ind, subgraph_score, obj_emb, fc_feats, att_masks = \
        self.gpn_layer(batchsize, num_features, num_relations, L,
                       sub_graph["sub_node_ind"], sub_graph["sub_edge_ind"],
                       sub_graph["sub_rel_ind"] , sub_graph["pool_mtx"], att_feats, x_pred,
                           fc_feats, sub_graph["att_masks"])


        if not self.using_semantic_graph:
            gpn_target = prompt2nodes(self.model, gt_graph["gt_nodes_name"], all_subgraph_obj_ind,
                                      pred_graph["image_nodes"], self.device,
                                      valid_threshold=self.valid_threshold, iou_threshold=self.iou_threshold,
                                      using_WN=self.using_WN,nodes_name = pred_graph["nodes_name"],nodes_range=self.max_nodes-1)
        else:
            gpn_target = prompt2nodes(self.model, gt_graph["gt_nodes_name"], all_subgraph_obj_ind,
                                      pred_graph["obj_nodes"][:,:self.max_nodes,:], self.device,
                                      valid_threshold=self.valid_threshold, iou_threshold=self.iou_threshold,
                                      using_WN=self.using_WN,nodes_name = pred_graph["nodes_name"],nodes_range=self.max_nodes-1)

        gpn_loss = self.GPN_loss(subgraph_score.view(-1).float().to(self.device), gpn_target.view(-1).float().to(self.device))


        # STEP-3 :
        # Graph-Graph matching
        if self.using_gt_graph_encoder:

            GT_obj_emb = self.GT_obj_emb_proj(gt_graph["gt_obj_nodes"])
            GT_obj_emb = self.relu(GT_obj_emb )
            # 把512的边文本特征（2,13,512)->（2,13,1024）
            GT_pred_fmap = self.GT_pred_emb_prj(gt_graph["gt_pred_emb"])
            gt_num_features = gt_graph["gt_obj_nodes"].size(1);
            gt_num_relations = gt_graph["gt_rel_ind"].size(1)
            GT_att_feats, GT_x_pred = self.GT_gcn_backbone(batchsize, gt_num_features, gt_num_relations, L,
                                                           GT_obj_emb, obj_dist, GT_pred_fmap,gt_graph["gt_rel_ind"])
            # 确保掩码是浮点数类型
            gt_nodes_mask = gt_graph["gt_nodes_mask"].float().unsqueeze(-1)  # 扩展掩码的最后一个维度
            GT_att_feats = GT_att_feats * gt_nodes_mask

            # 最大池化 (dim=1 表示在 num_nodes 维度上操作)
            max_pooled, _ = torch.max(GT_att_feats, dim=1)
            # 平均池化 (dim=1 表示在 num_nodes 维度上操作)
            mean_pooled = torch.mean(GT_att_feats, dim=1)

            # 堆叠结果 (dim=1 表示在特征维度上堆叠)
            GT_att_feats = torch.cat((max_pooled, mean_pooled), dim=1)
            GT_att_feats = self.gt_read_out_proj(GT_att_feats)

            score = self.mlp(fc_feats, GT_att_feats)
        # Graph-Text matching
        else :

            img_embedding = fc_feats.unsqueeze(1)
            #img_embedding = fc_feats
            text_features = self.txt_proj(gt_graph["text_features"])

            text_embedding = text_features.unsqueeze(1)
            tgt1 = self.self_attn(img_embedding, img_embedding, img_embedding)[0]
            tgt_v = img_embedding + self.dropout1(tgt1)
            tgt_v = self.norm1(tgt_v)
            tgt2 = self.self_attn(text_embedding, text_embedding, text_embedding)[0]
            tgt_l = text_embedding + self.dropout2(tgt2)
            tgt_l = self.norm2(tgt_l)
            attn_output_v, attn_output_l = self.bi_attention(tgt_v, tgt_l)
            enhance_feature_v = self.ffn_v(attn_output_v)
            enhance_feature_l = self.ffn_l(attn_output_l)
            score = self.mlp(enhance_feature_v,  enhance_feature_l)
            #score = self.mlp(img_embedding, text_features)


        return score,gpn_loss



if __name__ == '__main__':
    device = torch.device('cuda:0')
    #device = torch.device('cpu')
    model = OVSGTR_ALIGN(device)
    prompt = [
        "A bowl of soup that looks like a monster with tofu says deep learning",
        "a corgi",
    ]
    scene_graph=[
        {
        'nodes': [
            {'index': 0, 'name': 'food', 'location': [174.22494506835938, 58.40916442871094, 511.9583435058594, 457.85223388671875]},
            {'index': 1, 'name': 'plate', 'location': [109.30516052246094, -0.2391357421875, 511.8876953125, 511.2185363769531]},
            {'index': 2, 'name': 'table', 'location': [-0.01544189453125, -0.2085723876953125, 511.98419189453125, 511.7904052734375]},
            {'index': 3, 'name': 'bowl', 'location': [109.30516052246094, -0.2391357421875, 511.8876953125, 511.2185363769531]},
            {'index': 4, 'name': 'eye', 'location': [367.2624816894531, 210.0311279296875, 408.5400695800781, 251.5965576171875]},
            {'index': 5, 'name': 'eye', 'location': [322.8060607910156, 208.95352172851562, 364.5888977050781, 251.28009033203125]},
        ],
        'relations': [
            {'subject_name': 'plate', 'object_name': 'food', 'relation_name': 'of', 'subject_index1': 1, 'object_index2': 0, 'relation_score': 0.44991496205329895},
            {'subject_name': 'bowl', 'object_name': 'food', 'relation_name': 'of', 'subject_index1': 3, 'object_index2': 0, 'relation_score': 0.44991496205329895},
            {'subject_name': 'plate', 'object_name': 'table', 'relation_name': 'on', 'subject_index1': 1, 'object_index2': 2, 'relation_score': 0.4373536705970764},
            {'subject_name': 'bowl', 'object_name': 'table', 'relation_name': 'on', 'subject_index1': 3, 'object_index2': 2, 'relation_score': 0.4373536705970764},
            {'subject_name': 'plate', 'object_name': 'eye', 'relation_name': 'has', 'subject_index1': 1, 'object_index2': 4, 'relation_score': 0.32868239283561707},
            {'subject_name': 'bowl', 'object_name': 'eye', 'relation_name': 'has', 'subject_index1': 3, 'object_index2': 4, 'relation_score': 0.32868239283561707},
            {'subject_name': 'plate', 'object_name': 'eye', 'relation_name': 'has', 'subject_index1': 1, 'object_index2': 5, 'relation_score': 0.3265661597251892},
            {'subject_name': 'bowl', 'object_name': 'eye', 'relation_name': 'has', 'subject_index1': 3, 'object_index2': 5, 'relation_score': 0.3265661597251892},
            {'subject_name': 'plate', 'object_name': 'bowl', 'relation_name': 'on', 'subject_index1': 1, 'object_index2': 3, 'relation_score': 0.2953890562057495},
            {'subject_name': 'bowl', 'object_name': 'plate', 'relation_name': 'on', 'subject_index1': 3, 'object_index2': 1, 'relation_score': 0.2953890562057495},
            {'subject_name': 'food', 'object_name': 'eye', 'relation_name': 'has', 'subject_index1': 0, 'object_index2': 5, 'relation_score': 0.28586867451667786},
            {'subject_name': 'food', 'object_name': 'eye', 'relation_name': 'has', 'subject_index1': 0, 'object_index2': 4, 'relation_score': 0.2816210687160492},
        ],
    },
        {'nodes': [{'index': 0, 'name': 'dog',
                    'location': [128.55101013183594, 10.771820068359375, 465.95599365234375, 496.3731689453125]},
                   {'index': 1, 'name': 'head',
                    'location': [221.6790313720703, 13.391258239746094, 451.925537109375, 257.2298583984375]},
                   {'index': 2, 'name': 'ear',
                    'location': [225.5421142578125, 15.972755432128906, 299.2227783203125, 104.1164779663086]},
                   {'index': 3, 'name': 'mouth',
                    'location': [320.9255676269531, 223.4412384033203, 382.4273376464844, 249.3697967529297]},
                   {'index': 4, 'name': 'leg',
                    'location': [205.13848876953125, 351.8865051269531, 274.3372497558594, 492.8382873535156]},
                   {'index': 5, 'name': 'nose',
                    'location': [320.1405334472656, 200.84201049804688, 360.2065734863281, 236.093017578125]},
                   ],
         'relations': [{'subject_name': 'dog', 'object_name': 'nose', 'relation_name': 'has', 'subject_index1': 0,
                        'object_index2': 5, 'relation_score': 0.5209894776344299},
                       {'subject_name': 'dog', 'object_name': 'head', 'relation_name': 'has', 'subject_index1': 0,
                        'object_index2': 1, 'relation_score': 0.464683473110199},
                       {'subject_name': 'dog', 'object_name': 'leg', 'relation_name': 'on', 'subject_index1': 0,
                        'object_index2': 4, 'relation_score': 0.421752393245697},
                       {'subject_name': 'dog', 'object_name': 'mouth', 'relation_name': 'has', 'subject_index1': 0,
                        'object_index2': 3, 'relation_score': 0.38946959376335144},
                       {'subject_name': 'dog', 'object_name': 'ear', 'relation_name': 'has', 'subject_index1': 0,
                        'object_index2': 2, 'relation_score': 0.34158074855804443},
                       {'subject_name': 'leg', 'object_name': 'dog', 'relation_name': 'of', 'subject_index1': 4,
                        'object_index2': 0, 'relation_score': 0.2492642104625702},
                       {'subject_name': 'nose', 'object_name': 'leg', 'relation_name': 'on', 'subject_index1': 5,
                        'object_index2': 4, 'relation_score': 0.21383874118328094},
                       {'subject_name': 'leg', 'object_name': 'nose', 'relation_name': 'has', 'subject_index1': 4,
                        'object_index2': 5, 'relation_score': 0.20862995088100433},
                       {'subject_name': 'nose', 'object_name': 'head', 'relation_name': 'on', 'subject_index1': 5,
                        'object_index2': 1, 'relation_score': 0.2081722617149353},
                       {'subject_name': 'ear', 'object_name': 'head', 'relation_name': 'on', 'subject_index1': 2,
                        'object_index2': 1, 'relation_score': 0.20394504070281982},
                       {'subject_name': 'head', 'object_name': 'leg', 'relation_name': 'on', 'subject_index1': 1,
                        'object_index2': 4, 'relation_score': 0.20056675374507904},
                       {'subject_name': 'head', 'object_name': 'dog', 'relation_name': 'of', 'subject_index1': 1,
                        'object_index2': 0, 'relation_score': 0.19015191495418549},
                       ],
         },
    ]

    gt_graph = [
        {
            "nodes": [
                {"index": 0, "name": "bowl"},
                {"index": 1, "name": "soup"},
                {"index": 2, "name": "monster"},
                {"index": 3, "name": "tofu"},
                {"index": 4, "name": "deep learning"},
            ],
            "relations": [
                {"subject_name": "bowl", "object_name": "soup", "relation_name": "of", "subject_index1": 0,
                 "object_index2": 1},
                {"subject_name": "soup", "object_name": "monster", "relation_name": "looks like", "subject_index1": 1,
                 "object_index2": 2},
            ],
        },
        {
            "nodes": [
                {"index": 0, "name": "corgi"}
            ],
            "relations": [],
        },
    ]
    semantic_graph = [
        [
            {"gt_index": 0, "gt_name": "bowl", "node_index": 1, "node_name": "plate", "similarity": 0.8333333333333334,
             "relation_name": "RelatedTo", "relation_confidence": 5.6423399401312215},
            {"gt_index": 0, "gt_name": "bowl", "node_index": 2, "node_name": "table", "similarity": 0.7058823529411765,
             "relation_name": "AtLocation", "relation_confidence": 1.0},
            {"gt_index": 1, "gt_name": "soup", "node_index": 0, "node_name": "food", "similarity": 0.7692307692307693,
             "relation_name": "RelatedTo", "relation_confidence": 7.710771686413754},
            {"gt_index": 3, "gt_name": "tofu", "node_index": 0, "node_name": "food", "similarity": 0.7692307692307693,
             "relation_name": "RelatedTo", "relation_confidence": 1.0}
        ],
        [
            {"gt_index": 0, "gt_name": "corgi", "node_index": 0, "node_name": "dog", "similarity": 0.896551724137931,
             "relation_name": "IsA", "relation_confidence": 2.0}
        ]
    ]

    from PIL import Image
    import torchvision.transforms as transforms
    import torch

    # 图像路径列表
    image_paths = ['/home/ippl/LFY/TEXT2IMAGE/181.png', '/home/ippl/LFY/datasets/AIGCIQA2023/Image/allimg/2002.png']

    transform = transforms.Compose([transforms.ToTensor()])

    image = torch.stack([transform(Image.open(path)) for path in image_paths])

    score = model(image, prompt,scene_graph,gt_graph,semantic_graph)
    model.eval()
    print(score)