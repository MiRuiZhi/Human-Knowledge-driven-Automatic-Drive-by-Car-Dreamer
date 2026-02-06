import argparse
import datetime
import functools
import json
import os
import pickle
import shutil
import sys
import time
import uuid
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf

import dreamerv3
from dreamerv3 import embodied
from embodied.core import space  # 使用项目内部space模块

from dreamerv3 import ninjax as nj
from dreamerv3 import jaxutils
from dreamerv3.human_network import HumanNetwork
from dreamerv3.jax_data_load import JAXRSSMLoader
from dreamerv3.configs import get_default_human_network_config


def load_data(npz_path, batch_size, seq_length, seed=42):
    """
    加载训练数据的接口函数
    
    Args:
        npz_path: .npz数据文件路径
        batch_size: 批次大小
        seq_length: 序列长度
        seed: 随机种子
        
    Returns:
        loader: 数据加载器
        obs_shape: 观测空间
        act_shape: 动作空间
    """
    # 创建数据加载器
    loader = JAXRSSMLoader(npz_path, batch_size, seq_length, shuffle=True)

    # 获取观测和动作空间
    sample_data = np.load(npz_path)
    obs_shape = {'perception': space.Space(dtype=np.float32, shape=(26,))}  # 根据实际数据调整
    act_shape = space.Space(dtype=np.float32, shape=(3,))

    return loader, obs_shape, act_shape


def print_data_shapes(loader, seed=42, num_batches=3):
    """
    打印数据形状的接口函数
    
    Args:
        loader: 数据加载器
        seed: 随机种子
        num_batches: 要打印的批次数量
    """
    # 为每个epoch创建新的随机key
    rng = jax.random.PRNGKey(seed)
    
    batch_count = 0
    # 获取训练数据迭代器
    data_iter = loader.get_epoch_iterator(rng, 1)
    
    # 打印前几个batch的数据
    for batch_key, data_batch in data_iter:
        print(f"\nBatch {batch_count + 1}:")
        
        # 打印数据形状
        for key, value in data_batch.items():
            print(f"{key} shape: {value.shape}, dtype: {value.dtype}")
        
        batch_count += 1
        
        # 只打印前num_batches个batch
        if batch_count >= num_batches:
            break

    print(f"\nPrinted {batch_count} batches of data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', default="data/driving_dataset_rssm.npz", type=str, help='Path to training data NPZ file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--seq_length', type=int, default=50, help='Sequence length for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # 打印参数信息
    print(f"Loading data from: {args.npz_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")

    # 检查文件是否存在
    if not os.path.exists(args.npz_path):
        print(f"Error: Data file does not exist at {args.npz_path}")
        return

    # 加载数据
    loader, obs_shape, act_shape = load_data(args.npz_path, args.batch_size, args.seq_length, args.seed)
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {act_shape}")
    
    print("Data loading and printing first few batches...")
    print_data_shapes(loader, args.seed, 3)


if __name__ == '__main__':
    main()