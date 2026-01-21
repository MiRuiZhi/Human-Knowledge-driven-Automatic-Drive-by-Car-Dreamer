import jax
import jax.numpy as jnp

tree_map = jax.tree_util.tree_map
# 停止梯度传播的辅助函数，用于阻止某些计算路径上的梯度传播
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    """
    过滤日志中的类型检查消息
    """
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors, jaxagent, jaxutils, nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):
    """
    DreamerV3智能体主类
    结合了世界模型、任务行为策略和探索行为策略
    """
    def __init__(self, obs_space, act_space, step, config):
        """
        初始化智能体
        Args:
            obs_space: 观测空间
            act_space: 动作空间
            step: 步数计数器
            config: 配置对象
        """
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        # 世界模型：学习环境动态，包括编码器、RSSM和头部网络
        self.wm = WorldModel(obs_space, act_space, config, name="wm")
        # 任务行为策略：针对具体任务的策略
        self.task_behavior = getattr(behaviors, config.task_behavior)(self.wm, self.act_space, self.config, name="task_behavior")
        # 探索行为策略：用于探索环境的策略
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(self.wm, self.act_space, self.config, name="expl_behavior")

    def policy_initial(self, batch_size):
        """
        初始化策略所需的状态
        Args:
            batch_size: 批次大小
        Returns:
            包含世界模型、任务行为和探索行为初始状态的元组
        """
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        """
        初始化训练所需的状态
        Args:
            batch_size: 批次大小
        Returns:
            世界模型的初始状态
        """
        return self.wm.initial(batch_size)

    def policy(self, obs, state, mode="train"):
        """
        智能体策略函数：根据观测和当前状态生成动作
        Args:
            obs: 当前观测
            state: 当前状态
            mode: 执行模式（train, eval, explore）
        Returns:
            动作和新的状态
        """
        self.config.jax.jit and print("Tracing policy function.")
        obs = self.preprocess(obs)  # 预处理观测数据
        # 解析状态：(上一潜变量, 上一动作), 任务状态, 探索状态
        (prev_latent, prev_action), task_state, expl_state = state
        embed = self.wm.encoder(obs)  # 编码观测
        # 通过RSSM更新潜变量 latent为后验状态
        latent, _ = self.wm.rssm.obs_step(prev_latent, prev_action, embed, obs["is_first"])
        # 获取探索行为策略的输出
        self.expl_behavior.policy(latent, expl_state)
        task_outs, task_state = self.task_behavior.policy(latent, task_state)  # 任务行为策略
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)  # 探索行为策略
        
        # 根据执行模式选择输出
        if mode == "eval":
            outs = task_outs  # 评估模式使用任务行为策略
            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
        elif mode == "explore":
            outs = expl_outs  # 探索模式使用探索行为策略
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        elif mode == "train":
            outs = task_outs  # 训练模式使用任务行为策略
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        
        # 更新状态
        state = ((latent, outs["action"]), task_state, expl_state)
        return outs, state

    def train(self, data, state):
        """
        训练智能体：包括世界模型训练和策略优化
        Args:
            data: 训练数据
            state: 当前状态
        Returns:
            输出、新状态和训练指标
        """
        self.config.jax.jit and print("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)  # 预处理数据
        # 训练世界模型
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        # 构造上下文：合并原始数据和世界模型输出 原始数据键值对 和 世界模型键值对
        context = {**data, **wm_outs["post"]}
        # 重塑数据用于后续处理 不区分批次和时间维度
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
        # 训练任务行为策略  任务行为、探索行为  被Greedy装饰的policy网络
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        # 如果有探索行为策略，也进行训练
        if self.config.expl_behavior != "None":
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        # 如果数据包含优先级相关信息，准备输出
        if "keyA" in data.keys():
            outs = {
                "key": data["key"],
                "env_step": data["env_step"],
                "model_loss": metrics["model_loss_raw"].copy(),
                "td_error": metrics["td_error"].copy(),
            }
        else:
            outs = {}

        # 计算平均损失值用于报告
        metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        metrics.update({"td_error": metrics["td_error"].mean()})

        return outs, state, metrics

    def report(self, data):
        """
        生成训练报告
        Args:
            data: 用于报告的数据
        Returns:
            包含各种指标的字典
        """
        self.config.jax.jit and print("Tracing report function.")
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))  # 世界模型报告
        mets = self.task_behavior.report(data)  # 任务行为报告
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)  # 探索行为报告
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        """
        预处理观测数据
        Args:
            obs: 原始观测数据
        Returns:
            预处理后的观测数据
        """
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key", "env_step"):
                continue
            # 将uint8图像数据转换为float32并归一化到[0,1]
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        # 计算continuity信号（1-终止信号）
        obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        return obs


class WorldModel(nj.Module):
    """
    世界模型：学习环境的动态模型，用于预测未来状态
    包括编码器、RSSM（Recurrence State Space Model）和多个预测头
    """
    def __init__(self, obs_space, act_space, config):
        """
        初始化世界模型
        Args:
            obs_space: 观测空间
            act_space: 动作空间
            config: 配置对象
        """
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config
        # 获取观测空间的形状
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        # 创建多模态编码器
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name="enc")
        # RSSM：递归状态空间模型
        self.rssm = nets.RSSM(**config.rssm, name="rssm")
        # concept_bottleneck模块
        self.concept_bottleneck = nets.ConceptBottleneck(config.concept_bottleneck, name="concept")  # 应该会自己推理输入维度
        # 定义多个预测头：解码器、奖励、连续性
        # 为各个预测头添加concept作为输入
        concept_inputs = ['deter', 'stoch', 'alpha']  # 基础输入加上概念表示
        
        self.heads = {
            "decoder": nets.MultiDecoder(
                shapes, 
                inputs=concept_inputs, 
                **{k: v for k, v in config.decoder.items() if k != 'inputs'}, 
                name="dec"
            ),
            "reward": nets.MLP(
                (), 
                inputs=concept_inputs, 
                **{k: v for k, v in config.reward_head.items() if k != 'inputs'}, 
                name="rew"
            ),
            "cont": nets.MLP(
                (), 
                inputs=concept_inputs, 
                **{k: v for k, v in config.cont_head.items() if k != 'inputs'}, 
                name="cont"
            ),
        }
        # 世界模型优化器
        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)
        # 损失缩放配置, 给self.heads["decoder"].cnn_shapes里的键，mapping loss_scale里的值，生成一个字典
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales

    def initial(self, batch_size):
        """
        初始化世界模型状态
        Args:
            batch_size: 批次大小
        Returns:
            初始状态（潜变量和动作）
        """
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state):
        """
        训练世界模型
        Args:
            data: 训练数据
            state: 当前状态
        Returns:
            新状态、输出和指标
        """
        # 获取需要优化的模块
        modules = [self.encoder, self.rssm, *self.heads.values()]
        # 使用优化器训练模型
        mets, (state, outs, metrics) = self.opt(modules, self.loss, data, state, has_aux=True)
        metrics.update(mets)
        return state, outs, metrics

    def loss(self, data, state):
        """
        计算世界模型损失
        Args:
            data: 输入数据
            state: 当前状态
        Returns:
            损失值和相关指标
        """
        # 编码观测
        embed = self.encoder(data)  # B L d（被处理后的特征通道）
        prev_latent, prev_action = state  # 先前的潜变量和动作 RSSM 隐状态 和 动作 B Action维度  每个latend都有deter (B,512) stoch(B,32,32) logit(B,32,32)
        # 准备动作序列 prev_actions: B Length A                prev_action:  B*L a 
        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)
        # 通过RSSM观察得到后验和先验状态
        post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        # post： deter, stoch, logit  B L是分离的
        # prior： deter, stoch, logit  B L是分离的
        
        # 保存原始的post和prior用于概念瓶颈模块的损失计算
        orig_post, orig_prior = post, prior
        
        # 通过概念瓶颈模块处理post和prior
        post, prior, alpha = self.concept_bottleneck(orig_post, orig_prior)  # 被重建的post和prior，稀疏表示alpha

        dists = {}
        # 准备特征用于预测头  只用后验和原始编码观测
        feats = {**post, "embed": embed, "alpha": alpha}
        # feats： deter, stoch, logit, embed, alpha
        for name, head in self.heads.items():
            # 根据配置决定是否对特征停止梯度
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        
        losses = {}
        # 计算动态损失和表征损失 (使用经过概念瓶颈处理后的post和prior)
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        # 动态损失：告诉天气预报模型："你看，你预测的和实际很接近，继续保持"
        # 表征损失：告诉观测更新机制："不要因为一个异常天气就彻底改变认知"
        # 计算各预测头的损失
        for key, dist in dists.items():
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss
        
        # 计算概念瓶颈模块的损失（避免重复计算）
        concept_losses = self._compute_concept_losses(orig_post, orig_prior, post, prior, alpha)
        losses.update(concept_losses)
        
        # 应用损失缩放
        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        out = {"embed": embed, "post": post, "prior": prior, "alpha": alpha}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        # 更新状态
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action
        # 计算指标
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        # 存储原始模型损失用于优先经验回放
        metrics["model_loss_raw"] = model_loss
        return model_loss.mean(), (state, out, metrics)

    def imagine(self, policy, start, horizon):
        """
        在世界模型中进行想象（规划）
        Args:
            policy: 用于想象的策略
            start: 起始状态
            horizon: 想象的时域长度
        Returns:
            想象轨迹
        """
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)

        def step(prev, _):
            """
            想象步骤函数
            """
            prev = prev.copy()
            # 通过RSSM的图像步骤得到下一个状态
            state = self.rssm.img_step(prev, prev.pop("action"))
            return {**state, "action": policy(state)}

        # 扫描执行想象步骤
        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        # 计算连续性和折扣权重
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        """
        带携带信息的想象函数
        Args:
            policy: 用于想象的策略
            start: 起始状态
            horizon: 想象的时域长度
            carry: 携带信息
        Returns:
            想象轨迹
        """
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        outs, carry = policy(start, carry)
        start["action"] = outs
        start["carry"] = carry

        def step(prev, _):
            """
            带携带信息的想象步骤函数
            """
            prev = prev.copy()
            carry = prev.pop("carry")
            state = self.rssm.img_step(prev, prev.pop("action"))
            outs, carry = policy(state, carry)
            return {**state, "action": outs, "carry": carry}

        # 扫描执行想象步骤
        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items() if k != "carry"}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        """
        生成世界模型报告
        Args:
            data: 用于报告的数据
        Returns:
            包含报告指标的字典
        """
        state = self.initial(len(data["is_first"]))
        report = {}
        report.update(self.loss(data, state)[-1][-1])  # 计算并添加损失指标
        # 观察一小段数据用于重建和开放循环预测
        context, _ = self.rssm.observe(self.encoder(data)[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        start = {k: v[:, -1] for k, v in context.items()}
        # 重建和开放循环预测
        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](self.rssm.imagine(data["action"][:6, 5:], start))
        # 为解码器的CNN形状创建视频网格报告
        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(jnp.float32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils.video_grid(video)
        return report

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        """
        计算世界模型指标
        Args:
            data: 输入数据
            dists: 分布字典
            post: 后验状态
            prior: 先验状态
            losses: 损失字典
            model_loss: 总模型损失
        Returns:
            指标字典
        """
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils.tensorstats(entropy(prior), "prior_ent"))  # 先验熵
        metrics.update(jaxutils.tensorstats(entropy(post), "post_ent"))   # 后验熵
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})  # 损失均值
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})    # 损失标准差
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()
        metrics["reward_max_data"] = jnp.abs(data["reward"]).max()  # 奖励的最大绝对值
        metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()  # 预测奖励的最大绝对值
        # 如果有奖励分布，计算平衡统计
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        # 如果有连续性分布，计算平衡统计
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics

    def _compute_concept_losses(self, orig_post, orig_prior, post, prior, alpha):
        """
        计算概念瓶颈模块的损失
        Args:
            orig_post: 原始后验状态
            orig_prior: 原始先验状态
            post: 经过概念瓶颈处理后的后验状态
            prior: 经过概念瓶颈处理后的先验状态
            alpha: 稀疏表示
        Returns:
            概念模块损失字典
        """
        # 获取批次和时间维度，与其他损失保持一致
        batch_time_shape = list(orig_post.values())[0].shape[:2]  # [B, T]
        
        # 将所有状态合并以进行统一处理
        all_orig_values = list(orig_post.values()) + list(orig_prior.values())
        all_post_values = list(post.values()) + list(prior.values())
        
        # 计算重建损失 - 批量处理所有状态
        rec_losses = []
        for orig_val, post_val in zip(all_orig_values, all_post_values):
            # 对除了批次和时间维度以外的所有维度求平均，保持[B, T]维度
            if orig_val.ndim > 2:
                non_bt_dims = tuple(range(2, orig_val.ndim))
                squared_diff = (orig_val - post_val) ** 2
                # 添加数值稳定性处理
                rec_loss = jnp.clip(squared_diff, 0.0, 1e6).mean(axis=non_bt_dims)
            else:
                squared_diff = (orig_val - post_val) ** 2
                rec_loss = jnp.clip(squared_diff, 0.0, 1e6)
            rec_losses.append(rec_loss)
        
        # 平均所有键的重建损失
        rec_loss = sum(rec_losses) / len(rec_losses) if rec_losses else jnp.zeros(batch_time_shape)
        
        # 稀疏正则化损失: L1范数，保持[B, T]维度
        abs_alpha = jnp.clip(jnp.abs(alpha), 0.0, 1e6)  # 数值稳定性处理
        if alpha.ndim > 2:
            non_bt_dims = tuple(range(2, alpha.ndim))
            sparsity_loss = jnp.sum(abs_alpha, axis=non_bt_dims)
        else:
            sparsity_loss = abs_alpha
        
        # 计算KL散度损失，鼓励概念表示的多样性，保持[B, T]维度
        alpha_flat = alpha.reshape(alpha.shape[0], alpha.shape[1], -1)  # [B, T, -1]
        avg_alpha = jnp.mean(alpha_flat, axis=-1)  # [B, T]
        uniform_dist = jnp.ones_like(avg_alpha) / alpha_flat.shape[-1]
        # 添加数值稳定性处理
        diversity_loss = jnp.clip((avg_alpha - uniform_dist) ** 2, 0.0, 1e6)  # [B, T]
        
        # 获取稀疏正则化系数，如果配置中不存在则使用默认值
        lambda_reg = getattr(self.config, 'lambda_', 0.05)
        if hasattr(self.config, 'concept_bottleneck') and hasattr(self.config.concept_bottleneck, 'lambda_'):
            lambda_reg = self.config.concept_bottleneck.lambda_
        
        # 总概念损失，保持[B, T]维度
        total_concept_loss = rec_loss + lambda_reg * sparsity_loss + 0.1 * diversity_loss
        
        return {
            "concept_total_loss": total_concept_loss,
            "concept_reconstruction_loss": rec_loss,
            "concept_sparsity_loss": sparsity_loss,
            "concept_diversity_loss": diversity_loss
        }

    def imagine(self, policy, start, horizon):
        """
        在世界模型中进行想象（规划）
        Args:
            policy: 用于想象的策略
            start: 起始状态
            horizon: 想象的时域长度
        Returns:
            想象轨迹
        """
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)

        def step(prev, _):
            """
            想象步骤函数
            """
            prev = prev.copy()
            # 通过RSSM的图像步骤得到下一个状态
            state = self.rssm.img_step(prev, prev.pop("action"))
            return {**state, "action": policy(state)}

        # 扫描执行想象步骤
        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        # 计算连续性和折扣权重
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

class ImagActorCritic(nj.Module):
    """
    想象AC（Actor-Critic）：在世界模型中进行策略优化的演员-评论家算法
    使用想象轨迹来训练策略网络
    """
    def __init__(self, critics, scales, act_space, config):
        """
        初始化想象AC
        Args:
            critics: 评论家网络字典
            scales: 各项损失的缩放因子
            act_space: 动作空间
            config: 配置对象
        """
        # 过滤出缩放不为0的评论家
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        # 选择适当的梯度计算方法
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        # 创建演员网络（策略网络）
        self.actor = nets.MLP(
            name="actor",
            dims="deter",
            shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        # 为每个评论家创建回报归一化器
        self.retnorms = {k: jaxutils.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics}
        # 演员优化器
        self.opt = jaxutils.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        """
        初始化状态
        Args:
            batch_size: 批次大小
        Returns:
            初始状态
        """
        return {}

    def policy(self, state, carry):
        """
        策略函数
        Args:
            state: 当前状态
            carry: 携带信息
        Returns:
            包含动作的输出和新的携带信息
        """
        return {"action": self.actor(state)}, carry

    def train(self, imagine, start, context):
        """
        训练演员-评论家
        Args:
            imagine: 用于想象的函数
            start: 起始状态
            context: 上下文信息
        Returns:
            轨迹和训练指标
        """
        def loss(start):
            """
            损失函数：在想象轨迹上计算损失
            """
            # 创建策略函数，对状态停止梯度
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            # 生成想象轨迹
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            return loss, (traj, metrics)

        # 使用优化器训练演员网络
        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        # 训练所有评论家网络
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj, metrics

    def loss(self, traj):
        """
        计算演员网络损失
        Args:
            traj: 想象轨迹
        Returns:
            损失和指标
        """
        metrics = {}
        advs = []  # 优势值列表
        # 计算总缩放值
        total = sum(self.scales[k] for k in self.critics)
        # 为每个评论家计算优势
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)  # 计算奖励、回报和基线
            offset, invscale = self.retnorms[key](ret)  # 归一化回报
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            # 计算归一化的优势
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            # 记录各种统计信息
            metrics.update(jaxutils.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()

        # 计算TD误差用于优先经验回放
        r = jnp.reshape(rew[0], (self.config.batch_size, self.config.batch_length))
        v = jnp.reshape(base[0], (self.config.batch_size, self.config.batch_length))
        disc = jnp.reshape(traj["cont"][0], (self.config.batch_size, self.config.batch_length)) * (1 - 1 / self.config.horizon)
        td_error = r[:, :-1] + disc[:, 1:] * v[:, 1:] - v[:, :-1]
        metrics["td_error"] = td_error  # 存储TD误差用于PER优先级

        # 计算总体优势
        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        # 根据梯度类型选择损失函数
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent  # 减去熵正则项
        loss *= sg(traj["weight"])[:-1]  # 应用权重
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        """
        计算策略指标
        Args:
            traj: 轨迹
            policy: 策略
            logpi: 对数概率
            ent: 熵
            adv: 优势值
        Returns:
            指标字典
        """
        metrics = {}
        ent = policy.entropy()[:-1]
        # 计算随机性指标
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        # 处理离散或连续动作
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils.tensorstats(act, "action"))
        metrics.update(jaxutils.tensorstats(rand, "policy_randomness"))
        metrics.update(jaxutils.tensorstats(ent, "policy_entropy"))
        metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])  # 想象权重分布
        return metrics


class VFunction(nj.Module):
    """
    价值函数：使用神经网络近似价值函数
    """
    def __init__(self, rewfn, config):
        """
        初始化价值函数
        Args:
            rewfn: 奖励函数
            config: 配置对象
        """
        self.rewfn = rewfn
        self.config = config
        # 主网络和慢网络（用于稳定训练）
        self.net = nets.MLP((), name="net", dims="deter", **self.config.critic)
        self.slow = nets.MLP((), name="slow", dims="deter", **self.config.critic)
        # 慢网络更新器
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        # 评论家优化器
        self.opt = jaxutils.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        """
        训练价值函数
        Args:
            traj: 轨迹
            actor: 演员（策略）网络
        Returns:
            训练指标
        """
        # 计算目标值并停止梯度
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()  # 更新慢网络
        return metrics

    def loss(self, traj, target):
        """
        计算价值函数损失
        Args:
            traj: 轨迹
            target: 目标值
        Returns:
            损失和指标
        """
        metrics = {}
        # 截取轨迹（去除最后一个时间步）
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)  # 通过网络计算分布
        loss = -dist.log_prob(sg(target))  # 计算负对数似然损失
        # 应用慢正则化
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum("...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs))
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg  # 加上正则项
        loss = (loss * sg(traj["weight"])).mean()  # 应用权重并求平均
        loss *= self.config.loss_scales.critic  # 应用损失缩放
        metrics = jaxutils.tensorstats(dist.mean())  # 记录价值的统计信息
        return loss, metrics

    def score(self, traj, actor=None):
        """
        计算价值和回报
        Args:
            traj: 轨迹
            actor: 演员网络（可选）
        Returns:
            奖励、回报和价值
        """
        # 计算奖励
        rew = self.rewfn(traj)
        assert len(rew) == len(traj["action"]) - 1, "应该为除最后动作外的所有动作提供奖励"
        # 计算折扣因子
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        # 计算当前价值
        value = self.net(traj).mean()
        vals = [value[-1]]  # 从最后一个价值开始
        # 计算lambda回报
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]