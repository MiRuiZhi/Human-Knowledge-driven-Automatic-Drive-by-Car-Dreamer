import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from . import jaxutils
from . import ninjax as nj
from .nets import RSSM, Linear

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map


class DictionaryLearningLayer(nj.Module):
    """字典学习层，将输入映射到稀疏的概念原子表示"""
    def __init__(self, input_dim, num_concepts, concept_dim, sparsity_reg=1e-3, **kwargs):
        """
        Args:
            input_dim: 输入维度
            num_concepts: 概念数量（字典大小）
            concept_dim: 每个概念的维度
            sparsity_reg: 稀疏性正则化强度
        """
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.sparsity_reg = sparsity_reg
        self.kwargs = kwargs

    def __call__(self, x):
        """将输入x映射到稀疏的概念表示"""
        # 获取字典矩阵
        dictionary = self.get("dictionary", jnp.zeros, (self.num_concepts, self.concept_dim), f32)
        # 用tanh激活确保字典向量有界
        dictionary = jnp.tanh(dictionary)
        
        # 如果输入维度不等于概念维度，需要先投影
        if x.shape[-1] != self.concept_dim:
            x_projected = self.get("input_proj", Linear, self.concept_dim)(x)
        else:
            x_projected = x
        
        # 计算输入与字典中每个概念的相似度
        # x_projected: [..., concept_dim]
        # dictionary: [num_concepts, concept_dim]
        similarities = jnp.dot(x_projected, dictionary.T)  # [..., num_concepts]
        
        # 使用Softmax获得注意力权重，然后应用稀疏化
        # 使用soft-thresholding实现稀疏性
        threshold = self.get("threshold", jnp.zeros, (), f32)
        threshold = jax.nn.softplus(threshold)  # 确保阈值为正
        
        # 应用软阈值
        abs_similarities = jnp.abs(similarities)
        sparse_weights = jnp.sign(similarities) * jnp.maximum(abs_similarities - threshold, 0.0)
        
        # 返回稀疏权重和字典
        return sparse_weights, dictionary


class ConceptBottleneckRSSM(RSSM):
    """概念瓶颈RSSM模型，在RSSM后增加字典学习层"""
    def __init__(
        self,
        deter=1024,
        stoch=32,
        classes=32,
        unroll=False,
        initial="learned",
        unimix=0.01,
        action_clip=1.0,
        num_concepts=64,  # 新增概念数量
        concept_dim=None,  # 概念维度，默认为deter的一半
        sparsity_reg=1e-3,  # 稀疏性正则化强度
        **kw,
    ):
        super().__init__(deter, stoch, classes, unroll, initial, unimix, action_clip, **kw)
        
        # 设置概念瓶颈参数
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim or (deter // 2)  # 默认为deter的一半
        self.sparsity_reg = sparsity_reg
        
        # 创建字典学习层
        self.dict_layer = DictionaryLearningLayer(
            input_dim=deter,
            num_concepts=num_concepts,
            concept_dim=self.concept_dim,
            sparsity_reg=sparsity_reg
        )

    def get_concept_representation(self, state):
        """从RSSM状态获取概念表示"""
        # 提取确定性状态
        deter_state = state["deter"]
        
        # 通过字典学习层获得稀疏概念表示
        sparse_weights, dictionary = self.dict_layer(deter_state)
        
        # 返回稀疏权重作为概念表示
        return sparse_weights

    def obs_step(self, prev_state, prev_action, embed, is_first):
        """覆盖父类的obs_step方法，集成概念瓶颈"""
        is_first = jaxutils.cast_to_compute(is_first)
        prev_action = jaxutils.cast_to_compute(prev_action)
        if self._action_clip > 0.0:
            prev_action *= jaxutils.sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action)
        )
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
        
        # 通过字典学习层获得概念表示
        concept_repr = self.get_concept_representation(post)
        
        # 将概念表示添加到状态中
        post["concept"] = concept_repr
        
        return jaxutils.cast_to_compute(post), jaxutils.cast_to_compute(prior)

    def img_step(self, prev_state, prev_action):
        """覆盖父类的img_step方法，集成概念瓶颈"""
        prev_stoch = prev_state["stoch"]
        prev_action = jaxutils.cast_to_compute(prev_action)
        if self._action_clip > 0.0:
            prev_action *= jaxutils.sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
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
        
        # 通过字典学习层获得概念表示
        concept_repr = self.get_concept_representation(prior)
        
        # 将概念表示添加到状态中
        prior["concept"] = concept_repr
        
        return jaxutils.cast_to_compute(prior)

    def get_dist(self, state, argmax=False):
        """覆盖get_dist以支持概念状态"""
        if "concept" in state:
            # 如果状态中有概念表示，可以选择性地使用它
            pass
        
        return super().get_dist(state, argmax)

    def get_stoch(self, deter):
        """覆盖get_stoch以支持概念表示"""
        x = self.get("img_out", Linear, **self._kw)(deter)
        stats = self._stats("img_stats", x)
        dist = self.get_dist(stats)
        
        # 也可以为确定性状态生成概念表示
        concept_repr = self.dict_layer(deter)[0]  # 只取稀疏权重
        
        stoch_sample = dist.mode() if self._classes else dist.mean()
        return jaxutils.cast_to_compute({
            "stoch": stoch_sample,
            "deter": deter,
            **stats,
            "concept": concept_repr
        })