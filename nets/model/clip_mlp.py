import torch
import torch.nn as nn
from nets.encoder.clip import clip
import torch.nn.init as init
from nets.utils_net.prompts_split import extract_prompt,extract_attributes

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

'''
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化方法
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
'''


class CLIP_MLP(nn.Module):
    def __init__(self, ckpt, device):
        super(CLIP_MLP, self).__init__()
        self.device = device
        self.model, preprocess = clip.load(ckpt, device=device)

        # 创建MLP模型实例并确保在正确的设备上
        input_size = 512
        hidden_size = 256
        output_size = 1
        self.mlp = MLP(input_size, hidden_size, output_size).to(device)

        # 确保整个模型在指定设备上
        self.to(device)

    def forward(self, image, prompt):
        # 确保输入数据在正确的设备上
        image = image.to(self.device)
        token = clip.tokenize(prompt).to(self.device)

        if token.shape[0] != 1:
            token = token.squeeze()

        with torch.no_grad():
            text_features = self.model.encode_text(token)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            image_features = self.model.encode_image(image)

        score = self.mlp(image_features, text_features)
        return score

class CLIP_MLP_EXPERT(CLIP_MLP):
    def __init__(self, ckpt, device, task):
        super().__init__(ckpt, device)
        self.task = task


    def forward(self, image, prompt):
        # 使用 try-except 来处理 model 参数的合法性
        valid_models = {"des", "aes", "style"}
        if self.task not in valid_models:
            raise ValueError(f"Invalid model specified '{self.task}'. Valid models are: {valid_models}")

        # 提取 prompt 中的信息
        dict_prompt = extract_prompt(prompt)
        des, aes, style = extract_attributes(dict_prompt)

        # 根据 model 参数选择对应的 prompt
        selected_prompt = {
            "des": des,
            "aes": aes,
            "style": style
        }[self.task]
        print(selected_prompt)
        # 调用父类的 forward 方法
        expert_score = super().forward(image, selected_prompt)

        return expert_score


import torch
from torch.cuda import Event
import psutil
import time
import numpy as np


def measure_clip_mlp_performance(model, image, text, num_iterations=100):
    """
    测量 CLIP_MLP 模型的性能指标

    Args:
        model: CLIP_MLP模型实例
        image: 输入图像tensor
        text: 输入文本列表
        num_iterations: 重复测试的次数

    Returns:
        dict: 包含平均时间、标准差、GPU内存和CPU内存使用情况
    """
    # 初始化性能指标列表
    times = []

    # 初始化 CUDA events
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_memory = torch.cuda.memory_allocated()
        starter, ender = Event(enable_timing=True), Event(enable_timing=True)

    # 获取初始 CPU 内存
    initial_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # 预热
    print("Warming up...")
    with torch.inference_mode():
        for _ in range(10):
            _ = model(image, text)

    # 主测试循环
    print(f"Running {num_iterations} iterations...")
    with torch.inference_mode():
        for i in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                starter.record()

            start_time = time.time()

            # 运行推理
            _ = model(image, text)

            if torch.cuda.is_available():
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)  # 毫秒
            else:
                curr_time = (time.time() - start_time) * 1000  # 转换为毫秒

            times.append(curr_time)

            if (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{num_iterations} completed")

    # 计算性能指标
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    # 计算内存使用
    if torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated() - initial_gpu_memory
        peak_gpu_memory_mb = peak_gpu_memory / 1024 / 1024  # 转换为 MB
    else:
        peak_gpu_memory_mb = 0

    current_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    cpu_memory_used = current_cpu_memory - initial_cpu_memory

    # 打印结果
    print("\n" + "=" * 50)
    print("Performance Metrics:")
    print(f"Average Inference Time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"Peak GPU Memory Usage: {peak_gpu_memory_mb:.2f} MB")
    print(f"CPU Memory Usage: {cpu_memory_used:.2f} MB")
    print("=" * 50)

    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'peak_gpu_memory': peak_gpu_memory_mb,
        'cpu_memory_used': cpu_memory_used
    }


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CLIP_MLP(ckpt='/home/ippl/LFY/TEXT2IMAGE/model_data/ViT-B-32.pt', device=device)
    model.eval()  # 设置为评估模式

    text1 = "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains, casting long shadows across the peaceful valley as birds soar through the golden sky."
    text = [text1]
    image = torch.randn(1, 3, 224, 224).to(device)

    # 运行单次推理并打印结果
    score = model(image, text)
    print("Single inference result:", score, score.shape)

    # 运行性能测试
    metrics = measure_clip_mlp_performance(
        model=model,
        image=image,
        text=text,
        num_iterations=100  # 可以根据需要调整迭代次数
    )
