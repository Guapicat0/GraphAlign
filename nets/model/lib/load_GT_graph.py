import json


def load_gt_graphs(file_path):
    """从JSON文件加载所有的GT图形数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return []


def normalize_prompt(prompt):
    """标准化提示字符串，去除或统一处理特殊字符，特别处理引号和空白字符"""
    if not isinstance(prompt, str):
        return ""

    # 去除两端空白字符并转换为小写
    prompt = prompt.strip().lower()

    # 统一处理引号，将所有类型的引号替换为直引号 "
    prompt = prompt.replace('“', '"').replace('”', '"').replace('\'', '"')

    return prompt


def find_matching_graph(prompt, gt_graphs):
    """根据提示字符串在GT图形数据中找到最匹配的图形"""
    normalized_prompt = normalize_prompt(prompt)

    for graph in gt_graphs:
        if normalize_prompt(graph.get("prompt", "")) == normalized_prompt:
            return graph.get("GT_graph", {})

    return None


def get_matching_graphs_from_prompts(text_prompts, gt_graphs_file):
    """根据文本提示列表找到对应的GT图形，并返回仅包含nodes和relations的图形信息列表"""
    gt_graphs = load_gt_graphs(gt_graphs_file)
    matching_graphs = []

    for prompt in text_prompts:
        matching_graph = find_matching_graph(prompt, gt_graphs)
        if matching_graph and isinstance(matching_graph,
                                         dict) and 'nodes' in matching_graph and 'relations' in matching_graph:
            matching_graphs.append({
                'nodes': matching_graph['nodes'],
                'relations': matching_graph['relations']
            })
        else:
            print(f"Warning: No matching graph found for prompt: '{prompt}'. Adding empty graph.")
            matching_graphs.append({'nodes': [], 'relations': []})

    return matching_graphs