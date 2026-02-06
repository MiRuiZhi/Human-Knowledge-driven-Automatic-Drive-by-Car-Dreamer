import argparse
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp

import dreamerv3
from dreamerv3 import embodied
from dreamerv3 import ninjax as nj
from dreamerv3 import jaxutils
from dreamerv3.human_network import HumanNetwork
from dreamerv3.configs import get_default_human_network_config
from dreamerv3.jax_data_load import JAXRSSMLoader


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
    obs_shape = {'perception': embodied.Space(dtype=np.float32, shape=(26,))}  # 根据实际数据调整
    act_shape = embodied.Space(dtype=np.float32, shape=(3,))

    return loader, obs_shape, act_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', default="data/driving_dataset_rssm.npz", type=str, help='Path to training data NPZ file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--seq_length', type=int, default=50, help='Sequence length for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--logdir', type=str, default='/tmp/human_training', help='Logging directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # 打印参数信息
    print(f"Loading data from: {args.npz_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Log directory: {args.logdir}")

    # 检查文件是否存在
    if not os.path.exists(args.npz_path):
        print(f"Error: Data file does not exist at {args.npz_path}")
        return

    # 创建日志目录
    os.makedirs(args.logdir, exist_ok=True)

    # 使用load_data接口加载数据
    loader, obs_shape, act_shape = load_data(args.npz_path, args.batch_size, args.seq_length, args.seed)

    # 创建更平衡的配置，解决损失过大的问题
    config = embodied.Config({
        'encoder': {
            'mlp_keys': ['vector'],      # 需要编码的键，通常是向量观测
            'cnn_keys': r'^$',          # 正则表达式，禁用CNN处理（空正则）
            'mlp_layers': 3,           # 减少层数
            'mlp_units': 128,          # 减少单元数
            'mlp_act': 'silu',
            'mlp_norm': 'layer',
            'symlog_inputs': True,
        },
        
        # RSSM相关配置
        'rssm': {
            'deter': 256,              # 减少确定性状态维度
            'stoch': 32,               # 随机状态维度
            'classes': 32,
            'unroll': False,
            'initial': 'learned',
            'unimix': 0.01,
            'action_clip': 1.0,
            'units': 256,              # 减少单元数
        },
        
        # HumanNetwork相关配置
        'human_network': {
            'embed': 128,              # 减少嵌入维度
            'layers': 2,               # 减少层数
            'units': 256,              # 减少单元数
            'deter_size': 256,
            'stoch_size': 32,
            'categorical_size': 32,
            'action_layers': 2,        # 减少动作层
            'action_units': 256,       # 减少动作单元数
            'action_activation': 'silu',
            'action_norm': 'layer',
            'action_dist': 'trunc_normal',
            'action_temp': 0.1,
            'action_min_std': 0.1,
            'action_max_std': 1.0,
            'action_outscale': 1.0,
            'concept_dim': 64,         # 减少概念维度
            'dict_size': 64,           # 减少字典大小
            'lambda_sparse': 0.01,     # 减少稀疏性权重
        },
        
        # 概念模块配置
        'concept': {
            'inputs': ['embed', 'deter', 'stoch'],
            'concept_dim': 32,         # 减少概念维度
            'dict_size': 64,           # 减少字典大小
            'lambda_sparse': 0.01,
        },
        
        # 损失缩放配置
        'loss_scales': {
            'action_prediction': 1.0,   # 保持动作预测权重
            'dyn': 0.5,                # 降低动力学损失权重
            'rep': 0.1,                # 降低表征损失权重
            'concept_total_loss': 0.5, # 降低概念损失权重
            'concept_reconstruction_loss': 0.5,
            'concept_sparsity_loss': 0.1,
            'concept_diversity_loss': 0.1,
        },
        
        # RSSM损失配置
        'dyn_loss': {'impl': 'kl', 'free': 0.1},  # 降低自由项
        'rep_loss': {'impl': 'kl', 'free': 0.1},  # 降低自由项
        
        # 优化器配置
        'human_network_opt': {
            'opt': 'adam',
            'lr': 1e-4,               # 保持较小的学习率
            'eps': 1e-5,
            'clip': 1.0,              # 梯度裁剪，帮助稳定训练
            'wd': 0.0,
            'warmup': 0,
            'lateclip': 0.0,
        }
    })

    # 初始化网络
    print("Initializing HumanNetwork...")
    network = HumanNetwork(obs_shape, act_shape, config, name='human_network')
    
    # 初始化网络参数和状态
    print("Initializing network parameters...")
    rng = jax.random.PRNGKey(args.seed)
    batch_size = args.batch_size
    seq_len = args.seq_length

    # 创建初始状态
    def init_network():
        initial_state = network.initial(batch_size)
        return initial_state

    # 创建初始状态
    initial_state, _ = nj.pure(init_network)({}, rng)

    print(f"\nInitial state shapes:")
    print(f"  Deterministic state: {initial_state[0]['deter'].shape}")
    print(f"  Stochastic state: {initial_state[0]['stoch'].shape}")
    print(f"  Previous action: {initial_state[1].shape}")
    
    # 使用网络内部的优化器，不需要单独初始化
    print("Using internal network optimizer...")
    
    # 创建模拟数据用于初始化参数
    print("Creating dummy data for initialization...")
    dummy_data = {
        'perception': jnp.ones((batch_size, seq_len, 26), dtype=jnp.float32),
        'action': jnp.ones((batch_size, seq_len, 3), dtype=jnp.float32),
        'is_first': jnp.zeros((batch_size, seq_len), dtype=bool)
    }
    
    print("Initializing network parameters...")
    
    # 初始化参数 - 运行一次前向传播以初始化参数
    def init_params():
        action_pred, new_state, intermediates = network(dummy_data, initial_state)
        # 由于HumanNetwork没有params属性，我们不需要返回任何内容
        # 初始化已经在forward pass中完成
    
    # 执行初始化
    nj.pure(init_params)({}, rng)
    
    print("Starting training loop...")
    
    # 实际训练循环
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # 为每个epoch创建新的随机key
            rng = jax.random.fold_in(rng, epoch)
            
            batch_idx = 0
            # 获取训练数据迭代器
            data_iter = loader.get_epoch_iterator(rng, 1)
            
            epoch_loss = 0.0
            num_batches = 0
            total_batches = len(loader)  # 获取这个epoch的总batch数
            
            print(f"  Total batches in this epoch: {total_batches}")
            
            start_time = time.time()
            
            for batch_key, data_batch in data_iter:
                # 确保数据格式与HumanNetwork期望的格式匹配
                # 检查并调整数据键的名称
                adjusted_data_batch = {}
                
                # 将perception键改为perception_vector
                adjusted_data_batch['perception_vector'] = data_batch['perception']
                adjusted_data_batch['action'] = data_batch['action']
                adjusted_data_batch['is_first'] = data_batch['is_first']

                # 使用网络的初始状态进行训练
                train_state = initial_state
                
                # 执行单步训练 - 使用网络内部的train方法，需要用nj.pure包装
                def train_step():
                    return network.train(adjusted_data_batch, train_state)

                (new_state_batch, metrics), _ = nj.pure(train_step)({}, rng)
                
                # 使用安全的字典访问方式
                total_loss = metrics.get('total_loss', 0.0)
                epoch_loss += float(total_loss)  # 转换为Python标量以避免累积设备内存
                num_batches += 1
                
                # 每隔一定步数打印一次指标，增加打印频率
                if batch_idx % 10 == 0:  # 每10个batch打印一次
                    print(f"  Batch {batch_idx}/{total_batches}, Loss: {float(total_loss):.6f}, "
                          f"Action Loss: {float(metrics.get('action_loss', 0)):.6f}, "
                          f"Dyn Loss: {float(metrics.get('rssm_dyn_loss', 0)):.6f}, "
                          f"Rep Loss: {float(metrics.get('rssm_rep_loss', 0)):.6f}")
                
                batch_idx += 1
                    
            epoch_time = time.time() - start_time
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            print(f"  Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}, "
                  f"Batches processed: {num_batches}, Time: {epoch_time:.2f}s")
                    
    except StopIteration:
        print("Data iteration completed")
    
    # 保存最终模型
    model_path = os.path.join(args.logdir, "final_model.pkl")
    with open(model_path, 'wb') as f:
        # 保存网络参数，通过ninjax的机制获取
        def get_params():
            return network.params
        _, params = nj.pure(get_params)({}, rng)
        pickle.dump(params, f)  # 保存网络的参数
    print(f"Model saved to {model_path}")
    
    print("Training completed!")


if __name__ == '__main__':
    main()