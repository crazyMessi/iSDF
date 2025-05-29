import torch
import numpy as np
import json
import os

def load_dtype_config(config_path='config.json'):
    """加载配置文件中的数据类型设置"""
    if not os.path.exists(config_path):
        return {'numpy': 'float32', 'torch': 'float32'}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 如果配置文件中没有dtype设置，使用默认值
    if 'dtype' not in config:
        return {'numpy': 'float32', 'torch': 'float32'}
    
    return config['dtype']

def get_numpy_dtype(dtype_str=None):
    """获取numpy数据类型"""
    if dtype_str is None:
        dtype_config = load_dtype_config()
        dtype_str = dtype_config['numpy']
    
    # 将字符串转换为numpy数据类型
    if dtype_str == 'float16':
        return np.float16
    elif dtype_str == 'float32':
        return np.float32
    elif dtype_str == 'float64':
        return np.float64
    elif dtype_str == 'int32':
        return np.int32
    elif dtype_str == 'int64':
        return np.int64
    else:
        print(f"Warning: Unknown numpy dtype '{dtype_str}', using float32")
        return np.float32

def get_torch_dtype(dtype_str=None):
    """获取torch数据类型"""
    if dtype_str is None:
        dtype_config = load_dtype_config()
        dtype_str = dtype_config['torch']
    
    # 将字符串转换为torch数据类型
    if dtype_str == 'float16':
        return torch.float16
    elif dtype_str == 'float32':
        return torch.float32
    elif dtype_str == 'float64':
        return torch.float64
    elif dtype_str == 'int32':
        return torch.int32
    elif dtype_str == 'int64':
        return torch.int64
    else:
        print(f"Warning: Unknown torch dtype '{dtype_str}', using float32")
        return torch.float32

def numpy_to_tensor(arr, device=None):
    """将numpy数组转换为torch张量，使用配置的数据类型"""
    dtype_config = load_dtype_config()
    torch_dtype = get_torch_dtype(dtype_config['torch'])
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return torch.tensor(arr, dtype=torch_dtype, device=device)

def tensor_to_numpy(tensor):
    """将torch张量转换为numpy数组，使用配置的数据类型"""
    dtype_config = load_dtype_config()
    numpy_dtype = get_numpy_dtype(dtype_config['numpy'])
    
    return tensor.detach().cpu().numpy().astype(numpy_dtype) 