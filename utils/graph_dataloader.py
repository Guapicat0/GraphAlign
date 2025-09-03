from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms
from torchvision.transforms import Compose,ToTensor, Normalize, Resize
import torch
import numpy as np
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader

class SG_AIGCIQA2023(Dataset):
    def __init__(self, json_files, input_shape):
        """
        数据集类，读取图像路径、文本描述、gt_score，并加载图像。

        Args:
            json_files (str): JSON 文件路径。
            input_shape (tuple): 图像的输入尺寸 (height, width)。
        """
        self.data = []
        self.input_shape = input_shape

        # 加载 JSON 数据
        with open(json_files, 'r') as f:
            self.data = json.load(f)

        # 定义图像预处理
        self.transform = Compose([
            Resize((input_shape[0], input_shape[1])),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        # 读取图像路径
        img_path = entry['img_path']
        img = self.data_transform(img_path)

        # 提取文本描述和 GT 分数
        text_prompt = entry['prompt']
        gt_score = torch.tensor(float(entry['gt_score']), dtype=torch.float32).reshape(-1, 1)

        # 提取 scene_graph 和 GT_graph
        scene_graph = entry.get('scene_graph', None)
        gt_graph = entry.get('GT_graph', None)

        # 动态提取 semantic_graph，如果存在
        if 'semantic_graph' in entry:
            semantic_graph = entry['semantic_graph']
            return img, text_prompt, gt_score, scene_graph, gt_graph, semantic_graph

        # 返回图像张量、文本描述、GT 分数，以及 scene_graph 和 GT_graph
        return img, text_prompt, gt_score, scene_graph, gt_graph

    def data_transform(self, img_path):
        """
        图像读取和预处理函数。

        Args:
            img_path (str): 图像路径。

        Returns:
            torch.Tensor: 预处理后的图像张量。
        """
        img = Image.open(img_path).convert('RGB')
        #img = Image.new('RGB', (self.input_shape[0], self.input_shape[1]), color='black')
        img = self.transform(img)
        return img

class SG_AGIQA3K(Dataset):
    def __init__(self, json_files, input_shape):
        """
        数据集类，读取图像路径、文本描述、gt_score，并加载图像。

        Args:
            json_files (str): JSON 文件路径。
            input_shape (tuple): 图像的输入尺寸 (height, width)。
        """
        self.data = []
        self.input_shape = input_shape

        # 加载 JSON 数据
        with open(json_files, 'r') as f:
            self.data = json.load(f)

        # 定义图像预处理
        self.transform = Compose([
            Resize((input_shape[0], input_shape[1])),
            ToTensor(),
            #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        # 读取图像路径
        img_path = entry['img_path']
        img = self.data_transform(img_path)

        # 提取文本描述和 GT 分数
        text_prompt = entry['prompt']
        gt_score = torch.tensor(float(entry['gt_score']), dtype=torch.float32).reshape(-1, 1)

        # 提取 scene_graph 和 GT_graph
        scene_graph = entry.get('scene_graph', None)
        gt_graph = entry.get('GT_graph', None)

        # 动态提取 semantic_graph，如果存在
        if 'semantic_graph' in entry:
            semantic_graph = entry['semantic_graph']
            return img, text_prompt, gt_score, scene_graph, gt_graph, semantic_graph

        # 返回图像张量、文本描述、GT 分数，以及 scene_graph 和 GT_graph
        return img, text_prompt, gt_score, scene_graph, gt_graph

    def data_transform(self, img_path):
        """
        图像读取和预处理函数。

        Args:
            img_path (str): 图像路径。

        Returns:
            torch.Tensor: 预处理后的图像张量。
        """
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

class SG_RichHF18K(Dataset):
    def __init__(self, json_files, input_shape):
        """
        数据集类，读取图像路径、文本描述、gt_score，并加载图像。

        Args:
            json_files (str): JSON 文件路径。
            input_shape (tuple): 图像的输入尺寸 (height, width)。
        """
        self.data = []
        self.input_shape = input_shape

        # 加载 JSON 数据
        with open(json_files, 'r') as f:
            self.data = json.load(f)

        # 定义图像预处理
        self.transform = Compose([
            Resize((input_shape[0], input_shape[1])),
            ToTensor(),
            #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        # 读取图像路径
        img_path = entry['img_path']
        img = self.data_transform(img_path)

        # 提取文本描述和 GT 分数
        text_prompt = entry['prompt']
        gt_score = torch.tensor(float(entry['gt_score']), dtype=torch.float32).reshape(-1, 1)

        # 提取 scene_graph 和 GT_graph
        scene_graph = entry.get('scene_graph', None)
        gt_graph = entry.get('GT_graph', None)

        # 动态提取 semantic_graph，如果存在
        if 'semantic_graph' in entry:
            semantic_graph = entry['semantic_graph']
            return img, text_prompt, gt_score, scene_graph, gt_graph, semantic_graph

        # 返回图像张量、文本描述、GT 分数，以及 scene_graph 和 GT_graph
        return img, text_prompt, gt_score, scene_graph, gt_graph

    def data_transform(self, img_path):
        """
        图像读取和预处理函数。

        Args:
            img_path (str): 图像路径。

        Returns:
            torch.Tensor: 预处理后的图像张量。
        """
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

def dataset_collate(batch):
    """
    批处理函数，将 batch 中的数据合并为张量。

    Args:
        batch (list): 批量数据列表。

    Returns:
        torch.Tensor: 图像张量。
        list: 文本描述列表。
        torch.Tensor: GT 分数张量。
        list: scene_graph 列表。
        list: GT_graph 列表。
    """
    images, prompts, gt_scores, scene_graphs, gt_graphs = [], [], [], [], []
    semantic_graphs = []

    for entry in batch:
        # 判断 entry 的长度来动态解包
        if len(entry) == 6:  # 包含 semantic_graph
            img, prompt, gt_score, scene_graph, gt_graph, semantic_graph = entry
            semantic_graphs.append(semantic_graph)
        elif len(entry) == 5:  # 不包含 semantic_graph
            img, prompt, gt_score, scene_graph, gt_graph = entry
        else:
            raise ValueError(f"Unexpected entry structure with length {len(entry)}: {entry}")

        images.append(img)
        prompts.append(prompt)
        gt_scores.append(gt_score)
        scene_graphs.append(scene_graph)
        gt_graphs.append(gt_graph)

    images = torch.stack(images)  # 合并图像张量
    gt_scores = torch.cat(gt_scores)  # 合并 GT 分数

    # 如果 semantic_graphs 不为空，则返回它
    if semantic_graphs:
        return images, prompts, gt_scores, scene_graphs, gt_graphs, semantic_graphs

    # 如果 semantic_graphs 为空，则不返回它
    return images, prompts, gt_scores, scene_graphs, gt_graphs


def get_dataset_class(dataset_name):
    if dataset_name == 'SG-AIGCIQA2023':
        return SG_AIGCIQA2023
    if dataset_name == 'SG-AGIQA3K':
        return SG_AGIQA3K
    if dataset_name == 'SG-RichHF-18K':
        return SG_RichHF18K
    else:
        raise ValueError("Unsupported dataset name")


