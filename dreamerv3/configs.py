"""
HumanNetwork配置管理模块
提供预定义的配置函数，便于在训练脚本中使用
"""

import embodied


def make_config():
    """
    创建HumanNetwork的默认配置
    """
    # 基础配置
    config = embodied.Config({
        # Encoder相关配置 - 仅使用MLP处理向量数据
        'encoder': {
            'mlp_keys': ['vector'],      # 需要编码的键，通常是向量观测
            'cnn_keys': r'^$',          # 正则表达式，禁用CNN处理（空正则）
            'mlp_layers': 4,
            'mlp_units': 512,
            'mlp_act': 'silu',
            'mlp_norm': 'layer',
            'symlog_inputs': True,
        },
        
        # RSSM相关配置
        'rssm': {
            'deter': 512,
            'stoch': 32,
            'classes': 32,
            'unroll': False,
            'initial': 'learned',
            'unimix': 0.01,
            'action_clip': 1.0,
            'units': 512,
        },
        
        # HumanNetwork相关配置
        'human_network': {
            'embed': 512,              # 嵌入维度，即编码器输出维度
            'layers': 4,
            'units': 512,
            'deter_size': 512,
            'stoch_size': 32,
            'categorical_size': 32,
            'action_layers': 4,
            'action_units': 512,
            'action_activation': 'silu',
            'action_norm': 'layer',
            'action_dist': 'trunc_normal',
            'action_temp': 0.1,
            'action_min_std': 0.1,
            'action_max_std': 1.0,
            'action_outscale': 1.0,
            'concept_dim': 128,
            'dict_size': 128,
            'lambda_sparse': 0.05,
        },
        
        # 概念模块配置
        'concept': {
            'inputs': ['embed', 'deter', 'stoch'],
            'concept_dim': 128,
            'dict_size': 128,
            'lambda_sparse': 0.05,
        },
        
        # 损失缩放配置
        'loss_scales': {
            'action_prediction': 1.0,
            'dyn': 0.5,
            'rep': 0.1,
            'concept_total_loss': 1.0,
            'concept_reconstruction_loss': 1.0,
            'concept_sparsity_loss': 0.5,
            'concept_diversity_loss': 0.5,
        },
        
        # RSSM损失配置
        'dyn_loss': {
            'impl': 'kl',
            'free': 1.0,
        },
        'rep_loss': {
            'impl': 'kl',
            'free': 1.0,
        },
        
        # 优化器配置
        'human_network_opt': {
            'opt': 'adam',
            'lr': 1e-4,
            'eps': 1e-5,
            'clip': 100.0,
            'wd': 0.0,
            'warmup': 0,
            'lateclip': 0.0,
        }
    })
    
    return config


def make_custom_config(human_network_config=None, rssm_config=None, concept_config=None, loss_scale_config=None):
    """
    创建自定义配置
    
    Args:
        human_network_config: HumanNetwork的自定义配置
        rssm_config: RSSM的自定义配置
        concept_config: 概念模块的自定义配置
        loss_scale_config: 损失缩放的自定义配置
        
    Returns:
        配置对象
    """
    base_config = make_config()
    
    if human_network_config:
        base_config.update({'human_network': human_network_config})
    
    if rssm_config:
        base_config.update({'rssm': rssm_config})
    
    if concept_config:
        base_config.update({'concept': concept_config})
    
    if loss_scale_config:
        base_config.update({'loss_scales': loss_scale_config})
    
    return base_config


def get_default_human_network_config():
    """
    获取默认的HumanNetwork配置
    """
    return make_config()['human_network']


def get_default_rssm_config():
    """
    获取默认的RSSM配置
    """
    return make_config()['rssm']


def get_default_concept_config():
    """
    获取默认的概念模块配置
    """
    return make_config()['concept']


def get_default_loss_scale_config():
    """
    获取默认的损失缩放配置
    """
    return make_config()['loss_scales']