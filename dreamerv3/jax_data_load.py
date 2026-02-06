import argparse
import json
import os
import pickle
import shutil
import sys
import time
import uuid
from collections import defaultdict
from typing import Dict, Iterator, Optional, Tuple

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
from dreamerv3.configs import get_default_human_network_config


class JAXRSSMLoader:
    """JAX兼容的RSSM数据加载器，支持内存映射和随机采样，专为人类网络训练设计"""
    
    def __init__(self, npz_path: str, batch_size: int, seq_length: int = 50, shuffle: bool = True):
        """
        初始化JAX RSSM数据加载器
        
        Args:
            npz_path: .npz数据文件路径
            batch_size: 批次大小
            seq_length: 序列长度
            shuffle: 是否在每个epoch打乱数据
        """
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        
        # 内存映射方式加载npz文件，不将数据全部加载到内存
        self.data = np.load(npz_path, mmap_mode='r')
        
        # 获取数据维度和样本数量
        self.perceptions = self.data['perceptions']  # shape: [episodes, time_steps, perception_dim]
        self.actions = self.data['actions']  # shape: [episodes, time_steps, action_dim]
        self.num_episodes = self.perceptions.shape[0]
        self.total_time_steps = self.perceptions.shape[1]
        self.perception_dim = self.perceptions.shape[2]  # 26维感知
        self.action_dim = self.actions.shape[2]  # 3维动作
        
        # 验证数据一致性
        assert self.actions.shape[0] == self.num_episodes
        assert self.actions.shape[1] == self.total_time_steps
        
        # 计算可抽取的序列数量
        # 每个episode可以提取 (total_time_steps - seq_length + 1) 个序列
        self.sequences_per_episode = max(0, self.total_time_steps - self.seq_length + 1)
        self.num_sequences = self.num_episodes * self.sequences_per_episode
        
        print(f"Data loaded: {self.num_episodes} episodes, {self.total_time_steps} timesteps each")
        print(f"Perception dim: {self.perception_dim}, Action dim: {self.action_dim}")
        print(f"Total sequences available: {self.num_sequences}")

    def __len__(self) -> int:
        """返回每个epoch的批次数量"""
        return (self.num_sequences + self.batch_size - 1) // self.batch_size
    
    def get_sequence(self, episode_idx: int, time_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定episode和时间点的序列
        
        Args:
            episode_idx: 回合索引
            time_idx: 起始时间索引
            
        Returns:
            (perceptions_seq, actions_seq) - 特定长度的序列数据
        """
        perceptions_seq = self.perceptions[episode_idx, time_idx:time_idx+self.seq_length]
        actions_seq = self.actions[episode_idx, time_idx:time_idx+self.seq_length]
        return perceptions_seq, actions_seq
    
    def get_batch(self, key, batch_idx: Optional[int] = None) -> Dict[str, jnp.ndarray]:
        """
        获取一个随机批次的数据，适配人类网络输入格式
        
        Args:
            key: JAX随机数生成器key
            batch_idx: 可选的批次索引，如果提供则使用确定性采样
        
        Returns:
            data_dict - 符合人类网络输入格式的数据字典
        """
        if self.sequences_per_episode <= 0:
            raise ValueError(f"Sequence length ({self.seq_length}) is greater than available time steps ({self.total_time_steps})")
        
        # 随机选择批次中的每个序列
        key, subkey = jax.random.split(key)
        # 从所有可能的序列中采样
        seq_indices = jax.random.randint(
            subkey,
            shape=(self.batch_size,),
            minval=0,
            maxval=self.num_sequences
        )
        
        # 将全局序列索引转换为episode_idx和time_idx
        episode_indices = seq_indices // self.sequences_per_episode
        time_indices = seq_indices % self.sequences_per_episode
        
        # 批量提取序列
        batch_perceptions = []
        batch_actions = []
        
        for ep_idx, t_idx in zip(episode_indices, time_indices):
            percep_seq, action_seq = self.get_sequence(int(ep_idx), int(t_idx))
            batch_perceptions.append(percep_seq)
            batch_actions.append(action_seq)
        
        batch_perceptions = np.stack(batch_perceptions)
        batch_actions = np.stack(batch_actions)
        
        # 构造符合人类网络输入格式的数据字典
        data_dict = {
            'perception': jnp.array(batch_perceptions),  # [B, T, 26]
            'action': jnp.array(batch_actions),          # [B, T, 3]
            'is_first': jnp.zeros((self.batch_size, self.seq_length)),  # [B, T]
        }
        
        return data_dict
    
    def get_epoch_iterator(
        self, 
        key, 
        num_epochs: int = 1
    ) -> Iterator[Tuple[jax.Array, Dict[str, jnp.ndarray]]]:
        """
        获取一个epoch的数据迭代器
        
        Args:
            key: JAX随机数生成器key
            num_epochs: 要迭代的epoch数量
        
        Yields:
            (new_key, data_dict) - 新的随机key和批次数据
        """
        for epoch in range(num_epochs):
            num_batches = len(self)
            indices = np.arange(num_batches)
            
            if self.shuffle:
                key, subkey = jax.random.split(key)
                # 打乱批次索引
                indices = jax.random.permutation(subkey, indices)
            
            for i in indices:
                key, batch_key = jax.random.split(key)
                
                # 获取批次数据
                data_dict = self.get_batch(batch_key)
                
                yield batch_key, data_dict
    
    def get_full_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取完整的数据集（注意：这会将整个数据集加载到内存）
        
        Returns:
            (perceptions, actions) - 完整的数据集
        """
        return np.array(self.perceptions), np.array(self.actions)
    
    def close(self):
        """关闭内存映射文件"""
        self.data.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True, help='Path to training data NPZ file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--seq_length', type=int, default=50, help='Sequence length for training')
    args = parser.parse_args()

    # 打印参数信息
    print(f"Loading data from: {args.npz_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")

    # 检查文件是否存在
    if not os.path.exists(args.npz_path):
        print(f"Error: Data file does not exist at {args.npz_path}")
        return

    # 创建数据加载器
    loader = JAXRSSMLoader(args.npz_path, args.batch_size, args.seq_length, shuffle=True)
    
    # 测试数据加载
    rng = jax.random.PRNGKey(42)
    data_iter = loader.get_epoch_iterator(rng, 1)
    
    # 获取第一个批次的数据
    batch_key, data_batch = next(data_iter)
    
    print(f"\nSample batch shapes:")
    for key, value in data_batch.items():
        print(f"  {key}: {value.shape}")


if __name__ == '__main__':
    main()