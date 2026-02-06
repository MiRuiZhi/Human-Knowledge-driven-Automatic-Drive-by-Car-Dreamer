"""
训练 HumanNetwork 的脚本
使用假想数据训练 HumanNetwork，其中感知数据为 [B, T, D] 格式，动作数据为 [B, T, 3] 格式
"""

import argparse
import datetime
import functools
import os
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf

import dreamerv3
from dreamerv3 import embodied
from dreamerv3 import ninjax as nj
from dreamerv3 import jaxutils
from dreamerv3.human_network import HumanNetwork
from dreamerv3.configs import get_default_human_network_config


def main():
    args = parse_args()
    
    # 确保JAX使用可用的后端，而不是强制使用CPU
    # 如果需要强制使用CPU，可以取消下面一行的注释
    # jax.config.update('jax_platform_name', 'cpu')
    
    # 创建简单的观测和动作空间
    obs_space = {
        'perception_vector': embodied.Space(np.float32, (args.perception_dim,)),  # 26维感知向量
    }
    act_space = {'action': embodied.Space(np.float32, (3,))}  # 3维动作空间
    
    # 创建配置
    config = get_default_human_network_config()
    
    # 为概念模块和损失缩放添加配置
    config = config.update({
        'concept': {
            'inputs': ['embed', 'deter', 'stoch'],
            'concept_dim': 64,
            'dict_size': 128,
            'lambda_sparse': 0.01,
        },
        'loss_scales': {
            'action_prediction': 1.0,
            'dyn': 0.5,
            'rep': 0.1,
            'concept_total_loss': 1.0,
            'concept_reconstruction_loss': 1.0,
            'concept_sparsity_loss': 0.5,
            'concept_diversity_loss': 0.5,
        },
        'dyn_loss': {'impl': 'kl', 'free': 1.0},
        'rep_loss': {'impl': 'kl', 'free': 1.0},
    })
    
    print("="*60)
    print("开始测试 HumanNetwork...")
    print("创建 HumanNetwork 实例...")
    network = HumanNetwork(obs_space, act_space['action'], config, name='human_network')
    
    # 初始化网络参数和状态
    print("初始化 HumanNetwork...")
    rng = jax.random.PRNGKey(args.seed)
    batch_size = args.batch_size
    seq_len = args.seq_length

    # 在ninjax的pure环境中初始化
    def init_network():
        initial_state = network.initial(batch_size)
        return initial_state

    # 创建初始状态
    initial_state, _ = nj.pure(init_network)({}, rng)

    print(f"\n初始状态形状:")
    print(f"  确定性状态: {initial_state[0]['deter'].shape}")
    print(f"  随机状态: {initial_state[0]['stoch'].shape}")
    print(f"  上一动作: {initial_state[1].shape}")
    
    # 创建示例批次数据以适应你的任务
    # B, T, D 格式的感知数据
    perception_data = jnp.zeros((batch_size, seq_len, args.perception_dim))
    
    # B, T, 3 格式的动作数据（作为监督信号）
    action_data = jnp.zeros((batch_size, seq_len, 3))
    
    # 创建is_first标识，对于新序列，开始时标记为True
    is_first = jnp.zeros((batch_size, seq_len), dtype=bool)
    is_first = is_first.at[:, 0].set(True)  # 每个序列的第一个时间步为True
    
    # 将数据组织成字典格式
    data = {
        'perception_vector': perception_data,  # B, T, D
        'action': action_data,                 # B, T, 3
        'is_first': is_first                   # B, T
    }
    
    print(f"\n训练数据形状:")
    print(f"  感知向量: {data['perception_vector'].shape}")
    print(f"  动作 (监督): {data['action'].shape}")
    print(f"  是否开始 (序列重置): {data['is_first'].shape}")
    
    # 在pure环境中测试前向传播
    def forward_pass():
        # 使用网络的__call__方法进行前向传播，现在返回三个值
        action_pred, new_state, intermediates = network(data, initial_state)
        return action_pred, new_state, intermediates

    # 执行前向传播
    (action_pred, new_state, intermediates), _ = nj.pure(forward_pass)({}, rng)
    
    print(f"\n{'-'*20} 前向传播测试 {'-'*20}")
    print(f"前向传播成功完成!")
    print(f"预测动作分布类型: {type(action_pred)}")
    
    # 测试损失计算
    def compute_loss():
        total_loss, metrics = network.loss(data, initial_state, action_pred, intermediates)
        return total_loss, metrics

    result, _ = nj.pure(compute_loss)({}, rng)  # 第二次解包以获取状态
    total_loss, metrics = result  # 解包返回的元组
    
    print(f"\n{'-'*20} 损失计算测试 {'-'*20}")
    print(f"损失计算成功完成!")
    print(f"总损失: {total_loss}")
    # 测试训练步骤
    def train_step():
        new_state, metrics = network.train(data, initial_state)
        return new_state, metrics

    train_new_state, train_metrics = nj.pure(train_step)({}, rng)
    
    print(f"\n{'-'*20} 训练步骤测试 {'-'*20}")
    print(f"训练步骤成功完成!")
    print("Train metrics keys:", sorted([k for k in train_metrics.keys() if 'grad' not in k and 'step' not in k]))
    
    print(f"\n{'='*60}")
    print("HumanNetwork 测试完成!")


def parse_args():
    """解析命令行参数或使用默认值"""
    class Args:
        seed = 0
        batch_size = 8
        seq_length = 50
        perception_dim = 26
        action_dim = 3
        learning_rate = 3e-4
    
    return Args()


if __name__ == "__main__":
    main()