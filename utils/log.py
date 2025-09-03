import os
import json
import shutil


def evaluate_model_and_save_log(model_name, eval_score, path):
    # 读取现有的log.json文件内容
    log_file_path = os.path.join("logs", "log.json")

    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
        with open(log_file_path, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []
    else:
        log_data = []

    model_location = "nets/model/"
    source_model_path = model_name + ".py"
    global_source_model_path = os.path.join(model_location, source_model_path)
    destination_model_dir = os.path.dirname(path)

    # 使用 shutil.copy() 复制文件到目标目录
    shutil.copy(global_source_model_path, destination_model_dir)
    model_location = os.path.join(destination_model_dir, source_model_path)

    # 添加entry到log_data中
    entry = {
        "model": model_name,
        "computer_score": eval_score,
        "ckpt_path": path,
        "model_location": model_location
    }
    log_data.append(entry)

    # 保存更新后的log_data到log.json文件
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=4)

    print("log.json 文件已更新，并且 model.py 文件已复制到指定路径。")



