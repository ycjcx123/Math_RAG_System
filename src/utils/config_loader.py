import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

def inject_env_vars(config_dict):
    """
    递归遍历字典，如果发现 Key 对应的值在环境变量中存在，则替换。
    """
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # 如果值还是字典，继续往深了找
            inject_env_vars(value)
        else:
            # 核心逻辑：看这个 key 是不是我们想要覆盖的环境变量名
            # 比如 key 是 "DEEPSEEK_API_KEY"
            env_value = os.getenv(key)
            if env_value:
                config_dict[key] = env_value
    return config_dict

def load_config():
    # 1. 定位路径
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    config_path = BASE_DIR / "configs" / "config.yaml"
    env_path = BASE_DIR / ".env"

    # 2. 加载 .env 文件（如果存在）
    if env_path.exists():
        load_dotenv(env_path)

    # 3. 读取 YAML 原始配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 4. 覆盖逻辑：如果环境变量里有同名的 KEY，则覆盖
    config = inject_env_vars(config)

    return config