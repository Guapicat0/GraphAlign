import ast
import numpy as np
def split_prompt(prompts):

    if isinstance(prompts, np.ndarray):
        prompts = prompts.tolist()

    text_prompts = []
    scene_graphs = []

    for prompt in prompts:
        if not isinstance(prompt, str):
            raise ValueError("Each element in the list must be a string.")

        # 找到 scene_graph 开始的位置
        start_index = prompt.find("{'nodes':")

        if start_index == -1:
            raise ValueError("No scene_graph found in one of the prompts.")

        # 分割 prompt
        text_prompt = prompt[:start_index].strip()
        scene_graph = prompt[start_index:]

        text_prompts.append(text_prompt)
        scene_graphs.append(scene_graph)

    return text_prompts, scene_graphs

def process_scene_graphs(scene_graphs):
    nodes_list = []
    relations_list = []

    for scene_graph in scene_graphs:
        # 将字符串转换为字典
        #scene_graph= ast.literal_eval(scene_graph)

        # 提取nodes和relations
        nodes = scene_graph['nodes']
        relations = scene_graph['relations']

        # 将nodes和relations分别添加到对应的列表中
        nodes_list.append(nodes)
        relations_list.append(relations)

    return nodes_list, relations_list