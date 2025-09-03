from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize
import torch
import numpy as np
import scipy.io as sio
import os

class AGIQA3K(Dataset):
    def __init__(self, json_files, input_shape):
        self.data = []
        self.input_shape = input_shape
        #for json_file in json_files:
        with open(json_files, 'r') as f:
            entries = json.load(f)
            self.data.extend(entries)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        img_path = entry['img_path']
        prompt = entry['prompt']
        gt_score = torch.tensor(float(entry['gt_score']), dtype=torch.float32).reshape(-1, 1)
        img =self.data_transform(img_path , self.input_shape)

        # 返回图像tensor、prompt和gt_score
        return img, prompt, gt_score


    def data_transform(self, img_path,input_shape):
        # 在这里实现对图像的加载
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            Resize((input_shape[0], input_shape[1])),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        img = transform(img)

        return img

class AIGCIQA2023(Dataset):
    def __init__(self, json_files, input_shape):
        self.data = []
        self.input_shape = input_shape
        #for json_file in json_files:
        with open(json_files, 'r') as f:
            entries = json.load(f)
            self.data.extend(entries)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        img_path = entry['img_path']
        prompt = entry['prompt']
        gt_score = torch.tensor(float(entry['gt_score']), dtype=torch.float32).reshape(-1, 1)
        img =self.data_transform(img_path , self.input_shape)

        # 返回图像tensor、prompt和gt_score
        return img, prompt, gt_score


    def data_transform(self, img_path,input_shape):
        # 在这里实现对图像的加载
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            Resize((input_shape[0], input_shape[1])),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        img = transform(img)

        return img

def dataset_collate(batch):

    images, prompts, gt_scores = [], [], []

    for entry in batch:
        img, prompt, gt_score = entry

        images.append(img)
        prompts.append(prompt)
        gt_scores.append(gt_score)


    # 处理images
    if len(images) == 1:
        # batch_size=1的情况，直接添加batch维度
        images = images[0].unsqueeze(0)
        gt_scores = gt_scores[0].unsqueeze(0)
    else:
        # batch_size>1的情况，使用stack
        images = torch.stack(images)
        gt_scores = torch.cat(gt_scores)



    return images, prompts, gt_scores

def get_dataset_class(dataset_name):
    if dataset_name == 'AGIQA3K':
        return AGIQA3K
    elif dataset_name == 'AIGCIQA2023':
        return AIGCIQA2023
    else:
        raise ValueError("Unsupported dataset name")