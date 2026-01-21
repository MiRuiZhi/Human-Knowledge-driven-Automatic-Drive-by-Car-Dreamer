"""
神经网络模型定义文件

该文件定义了 DreamerV3 及相关模型使用的各种神经网络组件，
包括状态空间模型(RSSM)、编码器、解码器、全连接层、卷积层等基础组件，
以及用于可解释性的概念瓶颈层。
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from . import jaxutils
from . import ninjax as nj

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map


def sg(x):
    """
    停止梯度计算的函数，对输入的每个元素执行 stop_gradient 操作
    
    Args:
        x: 任意形状的张量或嵌套结构
    
    Returns:
        具有相同形状但停止梯度传播的张量
    """
    return tree_map(jax.lax.stop_gradient, x)


cast = jaxutils.cast_to_compute

class ConceptBottleneck(nj.Module):
    """
    概念瓶颈模块，用于提取可解释的概念表示
    实现类似LISTA（Learned ISTA）的稀疏编码机制
    """
    def __init__(
        self, 
        shapes,               # 输出形状字典
        config,               # 配置参数
        inputs=['deter'],  # 输入键列表
        knowledge_dim=128,    # 知识表示的维度
        enc_trans_ratio=2,    # 降维时，每次维度//2
        dec_trans_ratio=2,    # 升维时，每次维度*2
        activate='silu',      # 激活函数
        norm="none",          # 归一化方法
        n_atoms=128,          # 概念的数量
        lambda_=0.05,         # 稀疏正则化系数
        n_steps=50,           # ISTA迭代步数
        gamma=0.1,            # ISTA步长参数
        dims=None,           # 输入维度
        **kw                  # 其他关键字参数
    ):
        """
        初始化概念瓶颈模块
        
        Args:
            shapes: 输入形状字典
            inputs: 输入键列表
            excluded: 排除的字段
            knowledge_dim: 知识表示的维度
            enc_trans_ratio: 降维比例
            dec_trans_ratio: 升维比例
            activate: 激活函数
            norm: 归一化方法
            n_atoms: 字典原子数量
            lambda_: 稀疏正则化系数
            n_steps: ISTA迭代步数
            gamma: ISTA步长参数
            name: 模块名称
            **kw: 其他关键字参数
        """
        # 过滤形状
        excluded=("is_first", "is_last", "is_terminal", "reward"),  # 排除的字段
        shapes = {k: v for k, v in shapes.items() if k not in excluded}  # 要重建的输入特征

        self._inputs = Input(inputs, dims=dims)  # 创建输入处理器
        
        # 不管维度是多少，利用内置MLP，映射到knowledge_dim维
        self.encMLP = MLP((), **config.reward_head, name="rew")
        
        # 初始化字典矩阵 D ∈ ℝ^(n_atoms × knowledge_dim)
        self.D = nj.Variable(self._init_dict_matrix, name="dict_matrix")
        
        # 输入处理器
        self._inputs = Input(inputs)
    def compute(self, h):
        """
        LISTA前向传播函数
        
        Args:
            h: (B, knowledge_dim) - 编码后的特征
            
        Returns:
            tuple: (h_rec: (B, knowledge_dim) - 重建特征, alpha: (B, n_atoms) - 稀疏编码)
        """
        # 获取归一化的字典矩阵
        D = self.D.read()
        
        # 获取输入尺寸
        B, d = h.shape
        K = D.shape[0]
        
        # 初始化稀疏编码 α = 0
        alpha = jnp.zeros((B, K))
        
        # 预计算 Gram 矩阵 G = D^T @ D
        G = D @ D.T  # (K, K)
        
        # 预计算 D^T @ h
        Dth = h @ D.T  # (B, K)
        
        # ISTA 迭代过程
        def scan_fn(carry, _):
            alpha_t = carry
            
            # 计算梯度: ∇ = D^T @ (D @ α - h) = G @ α - D^T @ h
            grad = alpha_t @ G - Dth
            
            # ISTA 更新: α_{t+1} = soft_shrink(α_t - γ * ∇, λ * γ)
            next_alpha = self._soft_shrink(alpha_t - self.gamma * grad, self.lambda_ * self.gamma)
            
            return next_alpha, None
        
        # 执行ISTA迭代
        alpha_final, _ = jax.lax.scan(scan_fn, alpha, None, length=self.n_steps)
        
        # 重建: h_rec = α @ D
        h_rec = alpha_final @ D
        
        return h_rec, alpha_final

    def _soft_shrink(self, x, threshold):
        """
        Soft shrinkage 激活函数
        
        Args:
            x: 输入张量
            threshold: 阈值
            
        Returns:
            经过soft shrink变换的张量
        """
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)

    def loss(self, inputs):
        """
        计算概念瓶颈模块的损失
        
        Args:
            inputs: 输入特征字典
            
        Returns:
            tuple: (总损失, 损失详情字典)
        """
        reconstructed, alpha = self.__call__(inputs)
        
        # 将inputs也reshape以匹配
        features = self._inputs(inputs)
        flat_features = features.reshape([-1, features.shape[-1]])
        flat_reconstructed = reconstructed.reshape([-1, reconstructed.shape[-1]])

        # 重建损失: MSE
        rec_loss = jnp.mean((flat_features - flat_reconstructed) ** 2)
        
        # 稀疏正则化损失: L1范数
        sparsity_loss = jnp.sum(jnp.abs(alpha), axis=-1).mean()
        
        # 总损失
        total_loss = rec_loss + self.lambda_ * sparsity_loss
        
        return total_loss, {
            "rec_loss": rec_loss, 
            "sparsity_loss": sparsity_loss, 
            "alpha_norm": jnp.mean(jnp.abs(alpha))
        }

    def encode_decode(self, inputs):
        """
        编码解码过程，用于获取概念表示和重建
        
        Args:
            inputs: 输入特征字典
            
        Returns:
            tuple: (concept_repr: 概念表示, reconstructed: 重建特征)
        """
        reconstructed, alpha = self.__call__(inputs)
        return alpha, reconstructed


class RSSM(nj.Module):
    """
    循环状态空间模型 (Recurrent State Space Model)
    
    该模型结合了确定性和随机性状态，使用变分自编码器的思想来学习环境的潜在状态表示。
    它通过观察当前状态和动作来更新内部状态，并能够想象未来可能的状态序列。
    """
    def __init__(
        self,
        deter=1024,           # 确定性状态的维度
        stoch=32,             # 随机状态的维度
        classes=32,           # 随机状态的类别数
        unroll=False,         # 是否展开RNN循环
        initial="learned",    # 初始状态策略
        unimix=0.01,          # 均匀混合系数
        action_clip=1.0,      # 动作裁剪值
        **kw,                 # 其他关键字参数
    ):
        self._deter = deter
        self._stoch = stoch
        self._classes = classes
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._kw = kw

    def initial(self, bs):
        """
        初始化状态
        
        Args:
            bs: 批次大小
            
        Returns:
            初始状态字典
        """
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),     # 确定性状态
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),  # 对数概率
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),  # 随机状态
            )
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),     # 确定性状态
                mean=jnp.zeros([bs, self._stoch], f32),      # 均值
                std=jnp.ones([bs, self._stoch], f32),        # 标准差
                stoch=jnp.zeros([bs, self._stoch], f32),     # 随机状态
            )
        if self._initial == "zeros":
            return cast(state)
        elif self._initial == "learned":
            deter = self.get("initial", jnp.zeros, state["deter"][0].shape, f32)
            state["deter"] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state["stoch"] = self.get_stoch(cast(state["deter"]))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        """
        观察函数，根据嵌入、动作和是否是第一个时间步更新状态
        
        Args:
            embed: 观测嵌入
            action: 动作
            is_first: 是否是第一个时间步
            state: 当前状态
            
        Returns:
            后验状态和先验状态
        """
        def swap(x):
            return x.transpose([1, 0] + list(range(2, len(x.shape))))

        if state is None:
            state = self.initial(action.shape[0])

        def step(prev, inputs):
            return self.obs_step(prev[0], *inputs)

        inputs = swap(action), swap(embed), swap(is_first)
        start = state, state
        post, prior = jaxutils.scan(step, inputs, start, self._unroll)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        """
        想象函数，基于动作和初始状态预测未来状态序列
        
        Args:
            action: 动作序列
            state: 初始状态
            
        Returns:
            预测的状态序列
        """
        def swap(x):
            return x.transpose([1, 0] + list(range(2, len(x.shape))))

        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        prior = jaxutils.scan(self.img_step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_dist(self, state, argmax=False):
        """
        获取状态的概率分布
        
        Args:
            state: 当前状态
            argmax: 是否使用argmax而不是采样
            
        Returns:
            状态的概率分布
        """
        if self._classes:
            logit = state["logit"].astype(f32)
            return tfd.Independent(jaxutils.OneHotDist(logit), 1)
        else:
            mean = state["mean"].astype(f32)
            std = state["std"].astype(f32)
            return tfp.MultivariateNormalDiag(mean, std)

    def obs_step(self, prev_state, prev_action, embed, is_first):
        """
        观察步骤，在给定先前状态、动作和当前观测的情况下更新状态
        
        Args:
            prev_state: 前一个状态
            prev_action: 前一个动作
            embed: 当前观测嵌入
            is_first: 是否是第一个时间步
            
        Returns:
            后验状态和先验状态
        """
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
        prev_state, prev_action = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state,
            self.initial(len(is_first)),
        )
        prior = self.img_step(prev_state, prev_action)
        x = jnp.concatenate([prior["deter"], embed], -1)
        x = self.get("obs_out", Linear, **self._kw)(x)
        stats = self._stats("obs_stats", x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        """
        想象步骤，预测下一个状态而不使用真实观测
        
        Args:
            prev_state: 前一个状态
            prev_action: 前一个动作
            
        Returns:
            先验状态
        """
        prev_stoch = prev_state["stoch"]
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
        if self._classes:
            shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)
        x = jnp.concatenate([prev_stoch, prev_action], -1)
        x = self.get("img_in", Linear, **self._kw)(x)
        x, deter = self._gru(x, prev_state["deter"])
        x = self.get("img_out", Linear, **self._kw)(x)
        stats = self._stats("img_stats", x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        prior = {"stoch": stoch, "deter": deter, **stats}
        return cast(prior)

    def get_stoch(self, deter):
        """
        从确定性状态获取随机状态
        
        Args:
            deter: 确定性状态
            
        Returns:
            随机状态
        """
        x = self.get("img_out", Linear, **self._kw)(deter)
        stats = self._stats("img_stats", x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def _gru(self, x, deter):
        """
        GRU单元实现
        
        Args:
            x: 输入
            deter: 确定性状态
            
        Returns:
            输出和新的确定性状态
        """
        x = jnp.concatenate([deter, x], -1)
        kw = {**self._kw, "act": "none", "units": 3 * self._deter}
        x = self.get("gru", Linear, **kw)(x)
        reset, cand, update = jnp.split(x, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter, deter

    def _stats(self, name, x):
        """
        计算统计信息（均值、方差等）
        
        Args:
            name: 统计信息名称
            x: 输入
            
        Returns:
            统计信息字典
        """
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {"logit": logit}
            return stats
        else:
            x = self.get(name, Linear, 2 * self._stoch)(x)
            mean, std = jnp.split(x, 2, -1)
            std = 2 * jax.nn.sigmoid(std / 2) + 0.1
            return {"mean": mean, "std": std}

    def _mask(self, value, mask):
        """
        应用掩码到值上
        
        Args:
            value: 要应用掩码的值
            mask: 掩码
            
        Returns:
            应用掩码后的值
        """
        return jnp.einsum("b...,b->b...", value, mask.astype(value.dtype))

    def dyn_loss(self, post, prior, impl="kl", free=1.0):
        """
        计算动态损失
        
        Args:
            post: 后验状态
            prior: 先验状态
            impl: 损失计算方式
            free: 自由项
            
        Returns:
            动态损失
        """
        if impl == "kl":
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        elif impl == "logprob":
            loss = -self.get_dist(prior).log_prob(sg(post["stoch"]))
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def rep_loss(self, post, prior, impl="kl", free=1.0):
        """
        计算表征损失
        
        Args:
            post: 后验状态
            prior: 先验状态
            impl: 损失计算方式
            free: 自由项
            
        Returns:
            表征损失
        """
        if impl == "kl":
            loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
        elif impl == "uniform":
            uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
            loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
        elif impl == "entropy":
            loss = -self.get_dist(post).entropy()
        elif impl == "none":
            loss = jnp.zeros(post["deter"].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss


class MultiEncoder(nj.Module):
    """
    多模态编码器，处理CNN和MLP类型的输入数据
    
    将不同类型的输入（图像和向量）映射到统一的潜在空间
    """
    def __init__(
        self,
        shapes,               # 输入形状字典
        cnn_keys=r".*",       # 匹配CNN类型键的正则表达式
        mlp_keys=r".*",       # 匹配MLP类型键的正则表达式
        mlp_layers=4,         # MLP层数
        mlp_units=512,        # MLP每层单元数
        cnn="resize",         # CNN类型
        cnn_depth=48,         # CNN深度
        cnn_blocks=2,         # CNN块数
        resize="stride",      # 调整大小方式
        symlog_inputs=False,  # 是否对输入应用symlog变换
        minres=4,             # 最小分辨率
        **kw,                 # 其他关键字参数
    ):
        excluded = ("is_first", "is_last")
        shapes = {k: v for k, v in shapes.items() if (k not in excluded and not k.startswith("log_"))}
        self.cnn_shapes = {k: v for k, v in shapes.items() if (len(v) == 3 and re.match(cnn_keys, k))}
        self.mlp_shapes = {k: v for k, v in shapes.items() if (len(v) in (1, 2) and re.match(mlp_keys, k))}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)
        cnn_kw = {**kw, "minres": minres, "name": "cnn"}
        mlp_kw = {**kw, "symlog_inputs": symlog_inputs, "name": "mlp"}
        if cnn == "resnet":
            self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw)
        else:
            raise NotImplementedError(cnn)
        if self.mlp_shapes:
            self._mlp = MLP(None, mlp_layers, mlp_units, dist="none", **mlp_kw)

    def __call__(self, data):
        """
        对输入数据进行编码
        
        Args:
            data: 包含多个键值对的字典
            
        Returns:
            编码后的特征向量
        """
        some_key, some_shape = list(self.shapes.items())[0]  # 要观测的键和形状
        batch_dims = data[some_key].shape[: -len(some_shape)]  # B L
        data = {k: v.reshape((-1,) + v.shape[len(batch_dims) :]) for k, v in data.items()}  # key: np(B*L 原始观测的各自维度)
        outputs = []
        if self.cnn_shapes:
            inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)  # 拼接所有CNN输入  B*L 原始cnn拼接
            output = self._cnn(inputs)  # B*L d(8192)
            output = output.reshape((output.shape[0], -1))  # B*L d(8192) 由于我们只有一个图像观测，所以这里不需要分割
            outputs.append(output)
        if self.mlp_shapes:
            inputs = [data[k][..., None] if len(self.shapes[k]) == 0 else data[k] for k in self.mlp_shapes]
            inputs = jnp.concatenate([x.astype(f32) for x in inputs], -1)
            inputs = jaxutils.cast_to_compute(inputs)
            outputs.append(self._mlp(inputs))
        outputs = jnp.concatenate(outputs, -1)  # 操作：拼接所有的输出向量  B*L d(总的)
        outputs = outputs.reshape(batch_dims + outputs.shape[1:])  # 重塑为原来的形状 B L d(总的)
        return outputs


class MultiDecoder(nj.Module):
    """
    多模态解码器，将潜在表示解码回原始观测空间
    
    将统一的潜在表示解码为不同类型的输出（图像和向量）
    """
    def __init__(
        self,
        shapes,               # 输出形状字典
        inputs=["tensor"],    # 输入键列表
        cnn_keys=r".*",       # 匹配CNN类型键的正则表达式
        mlp_keys=r".*",       # 匹配MLP类型键的正则表达式
        mlp_layers=4,         # MLP层数
        mlp_units=512,        # MLP每层单元数
        cnn="resize",         # CNN类型
        cnn_depth=48,         # CNN深度
        cnn_blocks=2,         # CNN块数
        image_dist="mse",     # 图像分布类型
        vector_dist="mse",    # 向量分布类型
        resize="stride",      # 调整大小方式
        bins=255,             # 离散化箱数
        outscale=1.0,         # 输出缩放因子
        minres=4,             # 最小分辨率
        cnn_sigmoid=False,    # 是否在CNN输出应用sigmoid激活
        **kw,                 # 其他关键字参数
    ):
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3}
        self.mlp_shapes = {k: v for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}  # 确定要重建的输出shape
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)
        cnn_kw = {**kw, "minres": minres, "sigmoid": cnn_sigmoid}
        mlp_kw = {**kw, "dist": vector_dist, "outscale": outscale, "bins": bins}
        if self.cnn_shapes:
            shapes = list(self.cnn_shapes.values())
            assert all(x[:-1] == shapes[0][:-1] for x in shapes)  # 确保所有CNN输出的形状相同
            shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)  # 计算输出形状 128 128 3
            if cnn == "resnet":
                self._cnn = ImageDecoderResnet(shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name="cnn")
            else:
                raise NotImplementedError(cnn)
        if self.mlp_shapes:
            self._mlp = MLP(self.mlp_shapes, mlp_layers, mlp_units, **mlp_kw, name="mlp")
        self._inputs = Input(inputs, dims="deter")  # 将以 "deter" 键的张量作为维度参考标准
        self._image_dist = image_dist

    def __call__(self, inputs, drop_loss_indices=None):
        """
        解码输入特征到原始观测空间
        
        Args:
            inputs: 特征输入
            drop_loss_indices: 可选的损失索引列表，用于部分损失计算
            
        Returns:
            分布字典
        """
        features = self._inputs(inputs)  # 输入特征， B T d（被input组件处理过的， 1536）
        dists = {}  # 创建一个空的分布字典
        if self.cnn_shapes:
            feat = features
            if drop_loss_indices is not None:
                feat = feat[:, drop_loss_indices]
            flat = feat.reshape([-1, feat.shape[-1]])  # 合并B L维度 B*L d(1536)
            output = self._cnn(flat)  # B*L 128 128 3（取决于有多少个cnn观测和mlp观测）
            output = output.reshape(feat.shape[:-1] + output.shape[1:])  # B L 128 128 3
            split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])  # 获取每个CNN输出的索引
            means = jnp.split(output, split_indices, -1)  # 将输出分割成多个CNN输出
            dists.update({key: self._make_image_dist(key, mean) for (key, shape), mean in zip(self.cnn_shapes.items(), means)})  # 创建图像分布
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, name, mean):
        """
        创建图像分布
        
        Args:
            name: 分布名称
            mean: 均值
            
        Returns:
            图像分布对象
        """
        mean = mean.astype(f32)
        if self._image_dist == "normal":
            return tfd.Independent(tfd.Normal(mean, 1), 3)
        if self._image_dist == "mse":
            return jaxutils.MSEDist(mean, 3, "sum")
        raise NotImplementedError(self._image_dist)


class ImageEncoderResnet(nj.Module):
    """
    基于ResNet架构的图像编码器
    
    将图像转换为紧凑的特征表示，逐步降低空间分辨率并增加通道数
    """
    def __init__(self, depth, blocks, resize, minres, **kw):
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._kw = kw

    def __call__(self, x):
        """
        编码输入图像
        
        Args:
            x: 输入图像张量
            
        Returns:
            编码后的特征向量
        """
        stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
        depth = self._depth
        x = jaxutils.cast_to_compute(x) - 0.5
        # print(x.shape)
        for i in range(stages):
            kw = {**self._kw, "preact": False}
            if self._resize == "stride":
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, **kw)(x)
            elif self._resize == "stride3":
                s = 2 if i else 3
                k = 5 if i else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, **kw)(x)
            elif self._resize == "mean":
                N, H, W, D = x.shape
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
                x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
            elif self._resize == "max":
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
                x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "same")
            else:
                raise NotImplementedError(self._resize)
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, "preact": True}
                x = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw)(x)
                x = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw)(x)
                x += skip
                # print(x.shape)
            depth *= 2
        if self._blocks:
            x = get_act(self._kw["act"])(x)
        x = x.reshape((x.shape[0], -1))
        # print(x.shape)
        return x


class ImageDecoderResnet(nj.Module):
    """
    基于ResNet架构的图像解码器
    
    将紧凑的特征表示转换回图像，逐步增加空间分辨率并减少通道数
    """
    def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
        self._shape = shape
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._sigmoid = sigmoid
        self._kw = kw

    def __call__(self, x):
        """
        解码特征向量为图像
        
        Args:
            x: 输入特征向量
            
        Returns:
            解码后的图像张量
        """
        stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
        depth = self._depth * 2 ** (stages - 1)
        x = jaxutils.cast_to_compute(x)
        x = self.get("in", Linear, (self._minres, self._minres, depth))(x)  # 延迟初始化 指定输出维度，输入维度会自己计算
        for i in range(stages):
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, "preact": True}
                x = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw)(x)
                x = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw)(x)
                x += skip
                # print(x.shape)
            depth //= 2
            kw = {**self._kw, "preact": False}
            if i == stages - 1:
                kw = {}
                depth = self._shape[-1]
            if self._resize == "stride":
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, transp=True, **kw)(x)
            elif self._resize == "stride3":
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, transp=True, **kw)(x)
            elif self._resize == "resize":
                x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
            else:
                raise NotImplementedError(self._resize)
        if max(x.shape[1:-1]) > max(self._shape[:-1]):
            padh = (x.shape[1] - self._shape[0]) / 2
            padw = (x.shape[2] - self._shape[1]) / 2
            x = x[:, int(np.ceil(padh)) : -int(padh), :]
            x = x[:, :, int(np.ceil(padw)) : -int(padw)]
        # print(x.shape)
        assert x.shape[-3:] == self._shape, (x.shape, self._shape)
        if self._sigmoid:
            x = jax.nn.sigmoid(x)
        else:
            x = x + 0.5
        return x


class MLP(nj.Module):
    """
    多层感知机 (Multilayer Perceptron)
    
    一个灵活的全连接网络，可以输出标量、向量或分布
    """
    def __init__(
        self,
        shape,                # 输出形状
        layers,               # 层数
        units,                # 每层单元数
        inputs=["tensor"],    # 输入键列表
        dims=None,            # 维度参数
        symlog_inputs=False,  # 是否对输入应用symlog变换
        **kw,                 # 其他关键字参数
    ):
        assert shape is None or isinstance(shape, (int, tuple, dict)), shape
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self._layers = layers
        self._units = units
        self._inputs = Input(inputs, dims=dims)
        self._symlog_inputs = symlog_inputs
        distkeys = ("dist", "outscale", "minstd", "maxstd", "outnorm", "unimix", "bins")
        self._dense = {k: v for k, v in kw.items() if k not in distkeys}
        self._dist = {k: v for k, v in kw.items() if k in distkeys}

    def __call__(self, inputs):
        """
        对输入执行前向传播
        
        Args:
            inputs: 输入张量或字典
            
        Returns:
            网络输出  # dreamerv3.jaxutils.DiscDist obj
        """
        feat = self._inputs(inputs)  # B T d(input处理后的维度)
        if self._symlog_inputs:
            feat = jaxutils.symlog(feat)
        x = jaxutils.cast_to_compute(feat)  # 转换为计算类型 B T d不变
        x = x.reshape([-1, x.shape[-1]])  # B*T d不变
        for i in range(self._layers):
            x = self.get(f"h{i}", Linear, self._units, **self._dense)(x)  # 1536 -> 512 -> 512
        x = x.reshape(feat.shape[:-1] + (x.shape[-1],))  # B T 512
        if self._shape is None:
            return x
        elif isinstance(self._shape, tuple):
            return self._out("out", self._shape, x)
        elif isinstance(self._shape, dict):
            return {k: self._out(k, v, x) for k, v in self._shape.items()}
        else:
            raise ValueError(self._shape)

    def _out(self, name, shape, x):
        """
        生成输出层
        
        Args:
            name: 输出层名称
            shape: 输出形状
            x: 输入张量
            
        Returns:
            输出分布或张量
        """
        return self.get(f"dist_{name}", Dist, shape, **self._dist)(x)


class Dist(nj.Module):  # 自动根据shape生成对应的概率分布
    """
    分布层，生成指定类型的概率分布
    
    根据配置参数创建不同的概率分布，如正态分布、分类分布等
    """
    def __init__(
        self,
        shape,                # 输出形状
        dist="mse",           # 分布类型
        outscale=0.1,         # 输出缩放因子
        outnorm=False,        # 是否进行输出归一化
        minstd=1.0,           # 最小标准差
        maxstd=1.0,           # 最大标准差
        unimix=0.0,           # 均匀混合系数
        bins=255,             # 离散化箱数
    ):
        assert all(isinstance(dim, int) for dim in shape), shape
        self._shape = shape
        self._dist = dist
        self._minstd = minstd
        self._maxstd = maxstd
        self._unimix = unimix
        self._outscale = outscale
        self._outnorm = outnorm
        self._bins = bins

    def __call__(self, inputs):
        """
        生成概率分布
        
        Args:
            inputs: 输入张量
            
        Returns:
            概率分布对象
        """
        dist = self.inner(inputs)
        assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
            dist.batch_shape,
            dist.event_shape,
            inputs.shape,
        )
        return dist

    def inner(self, inputs):
        """
        内部实现，创建具体的分布
        
        Args:
            inputs: 输入张量
            
        Returns:
            概率分布对象
        """
        kw = {}
        kw["outscale"] = self._outscale
        kw["outnorm"] = self._outnorm
        shape = self._shape
        if self._dist.endswith("_disc"):
            shape = (*self._shape, self._bins)
        out = self.get("out", Linear, int(np.prod(shape)), **kw)(inputs)
        out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
        if self._dist in ("normal", "trunc_normal"):
            std = self.get("std", Linear, int(np.prod(self._shape)), **kw)(inputs)
            std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
        if self._dist == "symlog_mse":
            return jaxutils.SymlogDist(out, len(self._shape), "mse", "sum")
        if self._dist == "symlog_disc":
            return jaxutils.DiscDist(out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp)
        if self._dist == "mse":
            return jaxutils.MSEDist(out, len(self._shape), "sum")
        if self._dist == "normal":
            lo, hi = self._minstd, self._maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            dist = tfd.Normal(jnp.tanh(out), std)
            dist = tfd.Independent(dist, len(self._shape))
            dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "onehot":
            if self._unimix:
                probs = jax.nn.softmax(out, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                out = jnp.log(probs)
            dist = jaxutils.OneHotDist(out)
            if len(self._shape) > 1:
                dist = tfd.Independent(dist, len(self._shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
            return dist
        raise NotImplementedError(self._dist)


class Conv2D(nj.Module):
    """
    二维卷积层
    
    支持普通卷积和转置卷积，具有多种激活函数和归一化选项
    """
    def __init__(
        self,
        depth,                # 输出通道数
        kernel,               # 卷积核大小
        stride=1,             # 步长
        transp=False,         # 是否为转置卷积
        act="none",           # 激活函数
        norm="none",          # 归一化方法
        pad="same",           # 填充方式
        bias=True,            # 是否使用偏置
        preact=False,         # 是否预激活
        winit="uniform",      # 权重初始化方式
        fan="avg",            # fan模式
    ):
        self._depth = depth
        self._kernel = kernel
        self._stride = stride
        self._transp = transp
        self._act = get_act(act)
        self._norm = Norm(norm, name="norm")
        self._pad = pad.upper()
        self._bias = bias and (preact or norm == "none")
        self._preact = preact
        self._winit = winit
        self._fan = fan

    def __call__(self, hidden):
        """
        执行卷积操作
        
        Args:
            hidden: 输入特征图
            
        Returns:
            卷积后的特征图
        """
        if self._preact:
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
            hidden = self._layer(hidden)
        else:
            hidden = self._layer(hidden)
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
        return hidden

    def _layer(self, x):
        """
        实际的卷积层操作
        
        Args:
            x: 输入张量
            
        Returns:
            卷积后的张量
        """
        if self._transp:
            shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
            kernel = self.get("kernel", Initializer(self._winit, fan=self._fan), shape)
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_transpose(
                x,
                kernel,
                (self._stride, self._stride),
                self._pad,
                dimension_numbers=("NHWC", "HWOI", "NHWC"),
            )
        else:
            shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
            kernel = self.get("kernel", Initializer(self._winit, fan=self._fan), shape)
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_general_dilated(
                x,
                kernel,
                (self._stride, self._stride),
                self._pad,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
        if self._bias:
            bias = self.get("bias", jnp.zeros, self._depth, np.float32)
            bias = jaxutils.cast_to_compute(bias)
            x += bias
        return x


class Linear(nj.Module):
    """
    线性层（全连接层）
    
    实现基本的线性变换 y = xW + b，并支持归一化和激活函数
    """
    def __init__(
        self,
        units,                # 输出单元数
        act="none",           # 激活函数
        norm="none",          # 归一化方法
        bias=True,            # 是否使用偏置
        outscale=1.0,         # 输出缩放因子
        outnorm=False,        # 是否进行输出归一化
        winit="uniform",      # 权重初始化方式
        fan="avg",            # fan模式
    ):
        self._units = tuple(units) if hasattr(units, "__len__") else (units,)
        self._act = get_act(act)
        self._norm = norm
        self._bias = bias and norm == "none"
        self._outscale = outscale
        self._outnorm = outnorm
        self._winit = winit
        self._fan = fan

    def __call__(self, x):
        """
        执行线性变换
        
        Args:
            x: 输入张量
            
        Returns:
            变换后的张量
        """
        shape = (x.shape[-1], np.prod(self._units))
        kernel = self.get("kernel", Initializer(self._winit, self._outscale, fan=self._fan), shape)
        kernel = jaxutils.cast_to_compute(kernel)
        x = x @ kernel
        if self._bias:
            bias = self.get("bias", jnp.zeros, np.prod(self._units), np.float32)
            bias = jaxutils.cast_to_compute(bias)
            x += bias
        if len(self._units) > 1:
            x = x.reshape(x.shape[:-1] + self._units)
        x = self.get("norm", Norm, self._norm)(x)
        x = self._act(x)
        return x


class Norm(nj.Module):
    """
    归一化层
    
    提供多种归一化方法，目前只实现了层归一化
    """
    def __init__(self, impl):
        self._impl = impl

    def __call__(self, x):
        """
        执行归一化操作
        
        Args:
            x: 输入张量
            
        Returns:
            归一化后的张量
        """
        dtype = x.dtype
        if self._impl == "none":
            return x
        elif self._impl == "layer":
            x = x.astype(f32)
            x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
            x *= self.get("scale", jnp.ones, x.shape[-1], f32)
            x += self.get("bias", jnp.zeros, x.shape[-1], f32)
            return x.astype(dtype)
        else:
            raise NotImplementedError(self._impl)


class Input:
    """
    输入处理器
    
    将多个输入键合并为单个张量，用于处理多模态输入
    """
    def __init__(self, keys=["tensor"], dims=None):
        assert isinstance(keys, (list, tuple)), keys
        self._keys = tuple(keys)  # 拟提取的输入键
        self._dims = dims or self._keys[0]  # 拟确定的默认维度该服从哪个

    def __call__(self, inputs):
        """
        处理输入字典，将指定键的值连接起来
        
        Args:
            inputs: 输入字典或张量
            
        Returns:
            连接后的张量
        """
        if not isinstance(inputs, dict):
            inputs = {"tensor": inputs}
        inputs = inputs.copy()  # 安全拷贝
        for key in self._keys:
            if key.startswith("softmax_"):  # 若是softmax开头的键，则对对应的原始键应用softmax
                inputs[key] = jax.nn.softmax(inputs[key[len("softmax_") :]])
        if not all(k in inputs for k in self._keys):  # 确保所有的键都存在于输入字典中
            needs = f'{{{", ".join(self._keys)}}}'
            found = f'{{{", ".join(inputs.keys())}}}'
            raise KeyError(f"Cannot find keys {needs} among inputs {found}.")
        values = [inputs[k] for k in self._keys]  # 提取对应键的值
        dims = len(inputs[self._dims].shape)  # 获取目标张量的维度
        for i, value in enumerate(values):
            if len(value.shape) > dims:  # 若维度不一致(知识维度，不是每一个维度上的通道不一致)，则进行reshape 具体而言 输入张量的维度必须小于等于目标张量的维度
                values[i] = value.reshape(value.shape[: dims - 1] + (np.prod(value.shape[dims - 1 :]),))  # B T 32 32 -> B T 1024
        # B T 512
        # B T 1024
        values = [x.astype(inputs[self._dims].dtype) for x in values]
        # B T 512
        # B T 0124
        return jnp.concatenate(values, -1)  # B T 1536, 原来1536是这么来的，就是两个输入的通道数相加


class Initializer:
    """
    权重初始化器
    
    提供多种权重初始化方法，如均匀分布、正态分布和正交初始化
    """
    def __init__(self, dist="uniform", scale=1.0, fan="avg"):
        self.scale = scale
        self.dist = dist
        self.fan = fan

    def __call__(self, shape):
        """
        初始化权重张量
        
        Args:
            shape: 张量形状
            
        Returns:
            初始化后的权重张量
        """
        if self.scale == 0.0:
            value = jnp.zeros(shape, f32)
        elif self.dist == "uniform":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": (fanin + fanout) / 2, "in": fanin, "out": fanout}
            scale = self.scale / denoms[self.fan]
            limit = np.sqrt(3 * scale)
            value = jax.random.uniform(nj.rng(), shape, f32, -limit, limit)
        elif self.dist == "normal":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": np.mean((fanin, fanout)), "in": fanin, "out": fanout}
            scale = self.scale / denoms[self.fan]
            std = np.sqrt(scale) / 0.87962566103423978
            value = std * jax.random.truncated_normal(nj.rng(), -2, 2, shape, f32)
        elif self.dist == "ortho":
            nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
            matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
            mat = jax.random.normal(nj.rng(), matshape, f32)
            qmat, rmat = jnp.linalg.qr(mat)
            qmat *= jnp.sign(jnp.diag(rmat))
            qmat = qmat.T if nrows < ncols else qmat
            qmat = qmat.reshape(nrows, *shape[:-1])
            value = self.scale * jnp.moveaxis(qmat, 0, -1)
        else:
            raise NotImplementedError(self.dist)
        return value

    def _fans(self, shape):
        """
        计算输入和输出的扇入扇出数
        
        Args:
            shape: 张量形状
            
        Returns:
            (扇入数, 扇出数)
        """
        if len(shape) == 0:
            return 1, 1
        elif len(shape) == 1:
            return shape[0], shape[0]
        elif len(shape) == 2:
            return shape
        else:
            space = int(np.prod(shape[:-2]))
            return shape[-2] * space, shape[-1] * space


def get_act(name):
    """
    获取激活函数
    
    Args:
        name: 激活函数名称或可调用对象
        
    Returns:
        激活函数
    """
    if callable(name):
        return name
    elif name == "none":
        return lambda x: x
    elif name == "mish":
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    elif hasattr(jax.nn, name):
        return getattr(jax.nn, name)
    else:
        raise NotImplementedError(name)