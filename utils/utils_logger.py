import logging
import os
from datetime import datetime


def setup_logger(save_dir):
    # 创建logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # 如果logger已经有处理器，先清除
    if logger.handlers:
        logger.handlers.clear()

    # 创建固定的日志文件名
    log_file = os.path.join(save_dir, 'training.log')

    # 创建文件处理器，使用 'a' 模式进行追加写入
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger