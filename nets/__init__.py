from .model.clip_mlp import CLIP_MLP,CLIP_MLP_EXPERT
from .model.Graph_Align import OVSGTR_ALIGN


get_model_from_name = {
    "clip_mlp"        :   CLIP_MLP,
    "OVSGTR_ALIGN":OVSGTR_ALIGN,

}

class ModelSelector:
    """
    模型选择器，用于根据模型名称分别调用 IQA 模型和 Graph 模型。
    """
    def __init__(self):
        # 定义 IQA 模型和 Graph 模型的分组
        self.iqa_models = {
            "clip_mlp",
        }

        self.graph_models = {
            "OVSGTR_ALIGN"
        }

    def get_model(self, model_name, **kwargs):
        """
        根据名称获取模型。

        Args:
            model_name (str): 模型名称。
            **kwargs: 传递给模型的额外参数。

        Returns:
            object: 实例化后的模型对象。
        """
        if model_name in self.iqa_models:
            print(f"调用 IQA 模型: {model_name}")
            return get_model_from_name[model_name](**kwargs)

        elif model_name in self.graph_models:
            print(f"调用 Graph 模型: {model_name}")
            return get_model_from_name[model_name](**kwargs)

        else:
            raise ValueError(f"未找到模型名称 '{model_name}'，请检查输入。")

    def list_models(self):
        """
        列出所有支持的模型。

        Returns:
            dict: 包含 IQA 模型和 Graph 模型的名称。
        """
        return {
            "IQA Models": list(self.iqa_models),
            "Graph Models": list(self.graph_models),
        }

# 导出模型选择器
model_selector = ModelSelector()