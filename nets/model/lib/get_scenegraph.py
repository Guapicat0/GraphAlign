import torch
import torch.nn as nn
from PIL import Image
from nets.encoder.clip import clip
from nets.model.lib.SGG_FUNC import process_scene_graphs
from torchvision.transforms import functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision.transforms.functional import to_pil_image

class FeatureExtractor:
    def __init__(self, device,model,preprocess,max_nodes,max_relations,graph_node_dim,graph_edge_dim):
        super(FeatureExtractor, self).__init__()
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.max_nodes = max_nodes
        self.max_relations = max_relations
        self.graph_node_dim = graph_node_dim
        self.graph_edge_dim = graph_edge_dim

    def process_scene_graph(self, scene_graph):
        """
        Process the scene graph to extract nodes and relations.
        """
        nodes, relations = process_scene_graphs(scene_graph)
        return nodes, relations


    def extract_node_vision_features(self, image, nodes):
        """
        Extract visual features from the nodes in the image.
        """

        def show_cropped_image(cropped_image):
            """
            此函数用于将裁剪后的张量图像转换为 PIL 图像并显示
            :param cropped_image: 裁剪后的张量图像，形状为 (C, H, W)
            """
            # 将张量转换为 PIL 图像
            pil_image = to_pil_image(cropped_image)
            # 显示图像
            plt.imshow(pil_image)
            plt.axis('off')  # 关闭坐标轴
            plt.show()

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        #image = image.permute(0, 3, 1, 2)
        batch_size = image.size(0)
        cropped_tensors = []

        for i in range(batch_size):
            img = image[i]
            image_nodes = nodes[i]
            image_cropped_tensors = []

            # 异常处理
            names = [node['name'] for node in image_nodes]
            if not names:  # 如果 node_names 是空列表-->无节点图，graph必须含有节点
                raise ValueError(
                    f"Graphs dont allow 0-node, please examine your Graph dataset "
                )

            for node in image_nodes:
                location = node['location']
                x1, y1, x2, y2 = location
                cropped_image = F.crop(img, int(y1), int(x1), int(y2 - y1), int(x2 - x1))
                #show_cropped_image(cropped_image)
                cropped_image = transform(cropped_image)

                if cropped_image.dim() == 3:
                    cropped_image = cropped_image.unsqueeze(0)
                #cropped_image = self.preprocess(cropped_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    img_embedding = self.model.visual(cropped_image)
                    img_embedding /= img_embedding.norm(dim=-1, keepdim=True)
                image_cropped_tensors.append(img_embedding)

            Node_per_image_tensor = torch.stack(image_cropped_tensors, dim=0)

            current_node_length = Node_per_image_tensor.shape[0]

            if current_node_length > self.max_nodes:
                raise ValueError(
                    f"Graph at index {i} contains {current_node_length} nodes which exceeds the maximum allowed "
                    f"capacity of {self.max_nodes}. Please increase the maximum node capacity."
                )

            padded_features = torch.zeros((self.max_nodes, *Node_per_image_tensor.shape[1:]), device=self.device)
            padded_features[:current_node_length] = Node_per_image_tensor
            cropped_tensors.append(padded_features)

        att_feats = torch.stack(cropped_tensors, dim=0)

        # # add dummy node,in use of full fill tensor,因为有些子图并不是完整的图像，但是需要无效节点来填充张量
        # : relation +1 ,dection node +1
        att_feats = att_feats.squeeze()
        return att_feats

    def extract_node_textual_features(self, nodes):
        """
        Extract textual features from the node names.
        """
        batch_size = len(nodes)
        obj_tensors = []

        for i in range(batch_size):
            image_nodes = nodes[i]
            names = [node['name'] for node in image_nodes]

            if not names:  # 如果 node_names 是空列表-->无节点图，graph必须含有节点
                raise ValueError(
                    f"Graphs dont allow 0-node, please examine your Graph dataset "
                )

            token = clip.tokenize(names).to(self.device)
            if token.shape[0] != 1:
                token = token.squeeze()
            with torch.no_grad():
                text_features = self.model.encode_text(token).to(self.device)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

            current_node_length = text_features.shape[0]

            if current_node_length > self.max_nodes:
                raise ValueError(
                    f"Graph at index {i} contains {current_node_length} nodes which exceeds the maximum allowed "
                    f"capacity of {self.max_nodes}. Please increase the maximum node capacity."
                )

            padded_features = torch.zeros((self.max_nodes, text_features.shape[-1]), device=self.device)
            padded_features[:current_node_length] = text_features
            obj_tensors.append(padded_features)

        obj_nodes = torch.stack(obj_tensors, dim=0)


        return obj_nodes

    def extract_edge_textual_features(self, relations):
        """
        Extract textual features from the relation names.
        """
        batch_size = len(relations)
        pred_tensors = []

        for i in range(batch_size):
            image_relations = relations[i]
            edge_names = [rel['relation_name'] for rel in image_relations]

            if not edge_names:  # 如果 edge_names 是空列表--> 无边图
                edge_padded_features = torch.zeros((self.max_relations, self.graph_edge_dim), device=self.device)

            else:
                edge_token = clip.tokenize(edge_names).to(self.device)
                if edge_token.shape[0] != 1:
                    edge_token = edge_token.squeeze()
                with torch.no_grad():
                    edge_text_features = self.model.encode_text(edge_token).to(self.device)
                    edge_text_features = edge_text_features / edge_text_features.norm(dim=1, keepdim=True)

                current_relation_length = edge_text_features.shape[0]

                if current_relation_length > self.max_relations:
                    raise ValueError(
                        f"Graph at index {i} contains {current_relation_length} nodes which exceeds the maximum allowed "
                        f"capacity of {self.max_relations}. Please increase the maximum node capacity."
                    )

                edge_padded_features = torch.zeros((self.max_relations, edge_text_features.shape[-1]), device=self.device)
                edge_padded_features[:current_relation_length] = edge_text_features

            pred_tensors.append(edge_padded_features)

        pred_emb = torch.stack(pred_tensors, dim=0)

        return pred_emb

    def extract_relation_indices(self,relations):
        """
        Extract relation indices from the relations.
        """
        batch_size = len(relations)
        rel_ind = torch.full((batch_size, self.max_relations, 2), self.max_nodes - 1, dtype=torch.long,
                             device=self.device)
        for i, batch_relations in enumerate(relations):
            if not batch_relations:  # 如果关系列表为空
                continue  # 直接使用全填充的张量

            index_list = []
            for rel in batch_relations:
                index_pair = [rel['subject_index1'], rel['object_index2']]
                index_list.append(index_pair)

            index_tensor = torch.tensor(index_list, dtype=torch.long, device=self.device)

            current_relation_length = index_tensor.shape[0]

            if current_relation_length > self.max_relations:
                raise ValueError(
                    f"Graph at index {i} contains {current_relation_length} relations which exceeds the maximum allowed "
                    f"capacity of {self.max_relations}. Please increase the maximum relation capacity."
                )

            rel_ind[i, :current_relation_length, :] = index_tensor

        return rel_ind

    def generate_masks(self, nodes, relations):
        """
        Generate masks for nodes and edges in the batch of scene graphs.

        Parameters:
        - nodes: list of lists, where each sublist contains node dictionaries for a single graph.
        - relations: list of lists, where each sublist contains relation dictionaries for a single graph.

        Returns:
        - node_masks: Tensor of shape [batch_size, max_nodes], with 1s for real nodes and 0s elsewhere.
        - edge_masks: Tensor of shape [batch_size, max_relations], with 1s for real edges and 0s elsewhere.
        """
        batch_size = len(nodes)
        node_masks = torch.zeros((batch_size, self.max_nodes), dtype=torch.bool, device=self.device)
        edge_masks = torch.zeros((batch_size, self.max_relations), dtype=torch.bool, device=self.device)

        for i in range(batch_size):
            current_node_length = len(nodes[i])
            current_relation_length = len(relations[i])

            if current_node_length > self.max_nodes or current_relation_length > self.max_relations:
                raise ValueError(
                    f"Graph at index {i} exceeds the maximum allowed capacity. "
                    f"Nodes: {current_node_length}, Relations: {current_relation_length}. "
                    f"Please increase the maximum node/relation capacity."
                )

            node_masks[i, :current_node_length] = 1
            edge_masks[i, :current_relation_length] = 1

        return node_masks, edge_masks


