"""
HumanNetwork - 基于RSSM和概念学习的人类行为建模网络

该网络结合RSSM（Recurrent State Space Model）和概念学习模块，
用于学习和模拟人类驾驶员的行为模式。
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from dreamerv3 import nets, jaxutils, ninjax as nj
from dreamerv3.configs import make_config

# 设置JAX支持64位精度以避免警告
try:
    jax.config.update('jax_enable_x64', True)
except Exception:
    pass  # 如果无法设置x64，继续使用默认配置

tfd = tfp.substrates.jax.distributions

# 导入embodied
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import embodied

# 设置浮点类型 - 使用float32以确保兼容性
f32 = jnp.float32
f64 = jnp.float64
float_dtype = f32  # 使用f32以确保与优化器兼容


class HumanNetwork(nj.Module):
    """
    HumanNetwork类：结合RSSM和概念学习的人类行为建模网络
    """
    
    def __init__(self, obs_space, act_space, config=None, name='human_network'):
        """
        初始化HumanNetwork
        
        Args:
            obs_space: 观测空间
            act_space: 动作空间
            config: 配置对象
            name: 模块名称
        """
        # 先保存基本参数
        self.obs_space = obs_space
        self.act_space = act_space
        
        # 获取完整配置
        base_config = config or make_config()
        
        # 检查是否传入的是完整配置还是human_network部分
        if 'human_network' in base_config:
            # 传入的是完整配置
            self.config = base_config
            self.hn_config = self.config['human_network']
        else:
            # 传入的是human_network部分，需要构建完整配置
            self.config = make_config().update({'human_network': base_config})
            self.hn_config = base_config
        
        # 初始化各个组件
        self._setup_components()
    
    def _setup_components(self):
        """设置网络组件"""
        # 创建编码器：将原始观测数据投影到隐空间
        # 从配置中获取感知键
        self.perception_key = next((
            k for k in self.obs_space.keys() 
            if 'perception' in k or 'vector' in k or 
               (hasattr(self.obs_space[k], 'shape') and len(self.obs_space[k].shape) == 1 and self.obs_space[k].shape[0] == 26)
        ), 'vector')
        
        # 创建简单的MLP编码器，将输入投影到隐空间
        self.encoder = nets.MLP(
            256,  # 隐空间维度
            self.hn_config.action_layers,
            self.hn_config.action_units,
            ["embed"],  # 只有输入向量
            dims=None,
            act=self.hn_config.action_activation,
            norm=self.hn_config.action_norm,
            dist=self.hn_config.action_dist,
            temp=self.hn_config.action_temp,
            minstd=self.hn_config.action_min_std,
            maxstd=self.hn_config.action_max_std,
            outscale=self.hn_config.action_outscale,
            name="human_encoder"
        )
        
        # RSSM模型：处理时序依赖关系
        rssm_kwargs = dict(
            deter=self.hn_config.deter_size,
            units=self.hn_config.units,
            stoch=self.hn_config.stoch_size,
            classes=self.hn_config.categorical_size,
        )
        
        # 从配置中复制其他RSSM参数
        for key in ['unroll', 'initial', 'unimix', 'action_clip']:
            if hasattr(self.hn_config, key):
                rssm_kwargs[key] = getattr(self.hn_config, key)
        
        self.rssm = nets.RSSM(**rssm_kwargs, name='human_rssm')
        
        # 概念瓶颈模块：提取可解释的概念表示
        self.concept = nets.Concept(
            inputs=['embed', 'deter', 'stoch'],
            concept_dim=self.hn_config.concept_dim,
            dict_size=self.hn_config.dict_size,
            lambda_sparse=self.hn_config.lambda_sparse,
            name='human_concept'
        )
        
        # 动作预测头：从特征中预测连续动作
        self.action_head = nets.MLP(
            (int(np.prod(self.act_space.shape)),),  # 输出维度等于动作空间的形状
            self.hn_config.action_layers,
            self.hn_config.action_units,
            ["embed", "deter", "stoch", "alpha"],  # 包含概念特征
            dims=None,
            act=self.hn_config.action_activation,
            norm=self.hn_config.action_norm,
            dist=self.hn_config.action_dist,
            temp=self.hn_config.action_temp,
            minstd=self.hn_config.action_min_std,
            maxstd=self.hn_config.action_max_std,
            outscale=self.hn_config.action_outscale,
            name="human_action_head"
        )
        
        # 创建优化器
        self.opt = jaxutils.Optimizer(
            name='human_network_opt',
            opt=self.config.human_network_opt.opt,
            lr=self.config.human_network_opt.lr,
            eps=self.config.human_network_opt.eps,
            clip=self.config.human_network_opt.clip,
            wd=self.config.human_network_opt.wd,
            warmup=self.config.human_network_opt.warmup
        )

    def initial(self, batch_size):
        """
        初始化RSSM状态
        
        Args:
            batch_size: 批次大小
            
        Returns:
            初始状态元组，包含RSSM状态和初始动作
        """
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape), dtype=float_dtype)
        return prev_latent, prev_action

    def __call__(self, data, state):
        """
        前向传播

        Args:
            data: 包含观测和动作的数据字典
            state: 当前状态 (prev_latent, prev_action)

        Returns:
            tuple: (动作分布, 新状态, 中间结果字典)
        """
        # 解包状态
        prev_latent, prev_action = state
        
        # 提取感知数据
        perception_key = next((
            k for k in data.keys() 
            if 'perception' in k or 'vector' in k or 
               (len(data[k].shape) == 3 and data[k].shape[-1] == 26)
        ), None)
        
        # 如果没有找到特定的感知键，使用第一个非action/is_first/is_last/is_terminal键
        if perception_key is None:
            perception_key = next((
                k for k in data.keys() 
                if k not in ['action', 'is_first', 'is_last', 'is_terminal', 'reward', 'cont']
            ), None)
        
        if perception_key is None:
            raise ValueError("无法找到合适的感知数据键")
        
        # 通过MLP编码器获取嵌入表示
        perception = data[perception_key]
        encoded = self.encoder({'embed': perception})
        embed = encoded.mode().astype(float_dtype)
        
        # 准备动作序列
        prev_actions = jnp.concatenate([
            prev_action[:, None, :],
            data["action"][:, :-1, :]
        ], axis=1).astype(float_dtype)
        
        # 通过RSSM获取后验和先验状态
        post_state, prior_state = self.rssm.observe(
            embed, 
            prev_actions, 
            data.get("is_first", jnp.zeros((embed.shape[0], embed.shape[1]), dtype=jnp.bool_)), 
            prev_latent
        )
        
        # 整合当前特征
        feats = {**post_state, "embed": embed}
        
        # 通过概念模块处理
        concept_feats, alpha = self.concept(feats)
        concept_feats["alpha"] = alpha.astype(float_dtype)
        
        # 通过动作头预测动作
        action_dist = self.action_head(concept_feats)
        
        # 返回新状态（包括最后的动作）
        last_action = data["action"][:, -1, :].astype(float_dtype)
        new_state = (post_state, last_action)
        
        # 返回中间结果，供loss函数使用
        intermediates = {
            'embed': embed,
            'post_state': post_state,
            'prior_state': prior_state,
            'concept_feats': concept_feats,
            'alpha': alpha
        }
        
        return action_dist, new_state, intermediates

    def train(self, data, state):
        """
        训练一步
        Args:
            data: 训练数据
            state: 当前状态
        Returns:
            new_state: 更新后的状态
            metrics: 训练指标
        """
        # 定义损失函数，需要接收中间结果
        def loss_fn(data, state):
            # 执行前向传播获得中间结果
            action_pred, new_state, intermediates = self(data, state)
            # 计算损失，传入中间结果
            loss, metrics = self.loss(data, state, action_pred, intermediates)
            return loss, (new_state, metrics)

        # 执行训练步骤
        mets, (new_state, metrics) = self.opt([self.encoder, self.rssm, self.concept, self.action_head], 
                                              loss_fn, data, state, has_aux=True)
        metrics.update(mets)
        return new_state, metrics

    def loss(self, data, state, action_pred, intermediates):
        """
        计算损失：使用预测动作与真实动作的差异作为监督信号以及RSSM和Concept损失
        
        Args:
            data: 包含观测和真实动作的数据字典
            state: 当前状态
            action_pred: 预测的动作分布
            intermediates: __call__方法返回的中间结果
            
        Returns:
            tuple: (总损失, 指标字典)
        """
        # 计算动作预测损失
        action_true = data['action'].astype(float_dtype)
        action_logprob = action_pred.log_prob(action_true)
        
        # 防止log_prob产生极值，添加数值稳定性
        action_logprob = jnp.clip(action_logprob, -100, 100)
        action_loss = -jnp.mean(action_logprob)
        
        # 确保损失值为float32
        action_loss = action_loss.astype(jnp.float32)
        
        # 防止损失值过大，添加额外的数值稳定性
        action_loss = jnp.nan_to_num(action_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 从中间结果获取计算好的值
        post_state = intermediates['post_state']
        prior_state = intermediates['prior_state']
        alpha = intermediates['alpha']
        post_copy = post_state  # 为了concept loss
        
        # 计算RSSM损失 (dyn_loss 和 rep_loss)
        rssm_dyn_loss = self.rssm.dyn_loss(post_state, prior_state, **self.config.dyn_loss)
        rssm_rep_loss = self.rssm.rep_loss(post_state, prior_state, **self.config.rep_loss)
        
        # 确保RSSM损失为float32
        rssm_dyn_loss = rssm_dyn_loss.astype(jnp.float32)
        rssm_rep_loss = rssm_rep_loss.astype(jnp.float32)
        
        # 计算概念模块损失
        concept_loss_dict = self.concept.loss(post_copy, post_state, alpha)
        
        # 整合所有损失
        total_loss = (
            self.config.loss_scales.action_prediction * action_loss.astype(jnp.float32) +
            self.config.loss_scales.dyn * rssm_dyn_loss.mean().astype(jnp.float32) +
            self.config.loss_scales.rep * rssm_rep_loss.mean().astype(jnp.float32)
        )
        
        # 添加概念模块的损失
        for key, loss_value in concept_loss_dict.items():
            loss_value = loss_value.astype(jnp.float32)
            loss_scale_key = f"concept_{key}"
            if hasattr(self.config.loss_scales, loss_scale_key):
                total_loss += getattr(self.config.loss_scales, loss_scale_key) * loss_value.mean()
            elif hasattr(self.config.loss_scales, "concept_total_loss"):
                total_loss += getattr(self.config.loss_scales, "concept_total_loss") * loss_value.mean()
            else:
                # 默认情况下，如果没有明确配置概念损失缩放，则使用1.0
                total_loss += 1.0 * loss_value.mean()
        
        # 创建指标字典
        metrics = {
            'action_loss': action_loss,
            'rssm_dyn_loss': rssm_dyn_loss.mean(),
            'rssm_rep_loss': rssm_rep_loss.mean(),
            'total_loss': total_loss
        }
        
        # 添加概念模块指标
        for key, loss_value in concept_loss_dict.items():
            metrics[f'{key}'] = loss_value.mean()
        
        # 确保最终的总损失是标量float32
        total_loss = total_loss.astype(jnp.float32)
        
        # 添加梯度范数指标
        metrics['total_loss'] = total_loss
        
        return total_loss, metrics
    


class HumanNetworkWithFrozenParams:
    """
    封装冻结参数的人类网络，用于学生模型训练时的认知对齐
    """
    
    def __init__(self, human_network_params):
        self.params = human_network_params
    
    def get_frozen_action(self, data, state):
        """
        使用冻结参数获取人类动作预测
        Args:
            data: 观测数据
            state: 当前状态
        Returns:
            action: 预测的人类动作
        """
        # 这里需要通过jax.jit编译的函数来使用冻结参数
        # 具体实现将在实际训练中调用
        pass