from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
Sub-graph proposal network
"""
class gpn_layer(nn.Module):
    def __init__(self, GCN_dim=1024, hid_dim=512):
        super(gpn_layer, self).__init__()
        self.GCN_dim = GCN_dim

        # 增加更多的全连接层，以加深网络
        self.gpn_fc = nn.Sequential(nn.Linear(self.GCN_dim * 2, hid_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5),
                                    nn.Linear(hid_dim, 1),
                                    )
        nn.init.constant_(self.gpn_fc[0].bias, 0)
        nn.init.constant_(self.gpn_fc[3].bias, 0)


        self.gpn_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.read_out_proj = nn.Sequential(nn.Linear(self.GCN_dim*2, hid_dim),
                                           nn.Linear(hid_dim, self.GCN_dim*2))
        nn.init.constant_(self.read_out_proj[0].bias, 0)
        nn.init.constant_(self.read_out_proj[1].bias, 0)


    def forward(self,b,N,K,L,gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx,att_feats,x_pred,fc_feats,att_masks):
        """
        Input full graph, output sub-graph scores, sub-graph node features, and projected sub-graph read-out features
        extract sub-graph features --> pooling --> MLP --> sGPN score for each sub-graph, and index the sub-graphs with highest scores
        """
        # index subgraph node and edge features
        pos_obj_ind, neg_obj_ind, gpn_att, gpn_pred = self.extract_subgraph_feats(b,N,K,L, att_feats, gpn_obj_ind, x_pred, gpn_pred_ind)

        # max pooling and mean pooling
        read_out = self.graph_pooling(N, gpn_att, gpn_pool_mtx, att_masks)
        # MLP to get subgraph score
        subgraph_score = self.gpn_fc(read_out)  # pos, neg
        subgraph_score = self.sigmoid(subgraph_score) # pos, neg
        subgraph_score = subgraph_score.view(b,-1)
        sub_max_ind = torch.argmax(subgraph_score, dim=1).to(torch.long)


        all_subgraph_obj_ind = pos_obj_ind

        batch_ind = torch.arange(b).type_as(gpn_obj_ind).to(torch.long)
        sub_read_out = read_out.view(b,-1,read_out.size(-1))[batch_ind,sub_max_ind]
        fc_feats = self.read_out_proj(sub_read_out)


        subgraph_obj_ind = all_subgraph_obj_ind.cuda()[batch_ind.cuda(), sub_max_ind.cuda() , :].view(-1)
        att_feats = att_feats.cuda()[torch.arange(b).view(b, 1).expand(b, N).contiguous().view(-1).type_as(gpn_obj_ind),
                    subgraph_obj_ind, :].view(b, N, L)

        att_masks = att_masks[batch_ind, sub_max_ind, :]



        return sub_max_ind, all_subgraph_obj_ind, subgraph_score, att_feats, fc_feats, att_masks


    def extract_subgraph_feats(self, b,N,K,L, att_feats, gpn_obj_ind, x_pred, gpn_pred_ind):
        """
        Extract the node and edge features from full scene graph by using the sub-graph indices.
        """
        # index subgraph object and predicate features
        pos_obj_ind = gpn_obj_ind
        obj_batch_ind = torch.arange(b).view(b,1).expand(b,N*gpn_obj_ind.size(-2)).contiguous().view(-1).type_as(gpn_obj_ind)
        # 确保索引张量是 long 类型
        pos_obj_ind = pos_obj_ind.to(torch.long)
        obj_batch_ind = obj_batch_ind.to(torch.long)
        pos_gpn_att = att_feats[obj_batch_ind, pos_obj_ind.contiguous().view(-1)]


        pos_pred_ind = gpn_pred_ind.contiguous().view(-1)
        pred_batch_ind = torch.arange(b).view(b,1).expand(b,K*gpn_pred_ind.size(-2)).contiguous().view(-1).type_as(gpn_pred_ind)
        pred_batch_ind = pred_batch_ind.to(torch.long)

        pos_pred_ind = pos_pred_ind.to(torch.long)
        gpn_att = pos_gpn_att.view(-1, N, L)  # pos, neg
        if x_pred is not None:
            pos_gpn_pred = x_pred[pred_batch_ind, pos_pred_ind]
            gpn_pred = pos_gpn_pred.view(-1,K,L)
        else:
            gpn_pred = None

        neg_obj_ind=None

        return pos_obj_ind, neg_obj_ind, gpn_att, gpn_pred

    def graph_pooling(self, N, gpn_att, gpn_pool_mtx, att_masks):
        """
        Pooling features over nodes of input sub-graphs.
        """
        # batch-wise max pooling and mean pooling, by diagonal matrix
        each_pool_mtx = torch.transpose(gpn_pool_mtx, 0, 1).contiguous().view(-1,N,N) 
        clean_feats = torch.bmm(each_pool_mtx,gpn_att)
        '''
        max_feat = torch.max(clean_feats,dim=1)[0]
        mean_feat = torch.sum(clean_feats,dim=1) / torch.transpose(att_masks,0,1).sum(-1).view(-1,1)
        read_out = torch.cat((max_feat, mean_feat),dim=-1) 
        '''
        # 定义最大池化层
        max_pool = nn.MaxPool1d(kernel_size=clean_feats.size(1))

        # 将 clean_feats 的形状调整为 (128, 1024, 7)，因为 MaxPool1d 需要输入的形状为 (N, C, L)
        clean_feats_transposed = clean_feats.transpose(1, 2)  # (128, 1024, 7)

        # 进行最大池化
        max_pooled = max_pool(clean_feats_transposed)

        # 将结果形状调整回 (128, 1024)
        max_pooled = max_pooled.squeeze(2)

        avg_pooled = clean_feats.mean(dim=1)
        read_out = torch.cat((max_pooled, avg_pooled), dim=-1)
        return read_out



