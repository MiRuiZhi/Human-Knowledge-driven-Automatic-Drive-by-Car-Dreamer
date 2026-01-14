"""
DreamerV3训练循环实现
这个脚本定义了训练的主要逻辑，包括环境交互、数据收集、模型训练和日志记录
"""
import re

import embodied
import jax
import numpy as np


def train(agent, env, replay, logger, args):
    """
    DreamerV3的主要训练函数
    参数:
        agent: 智能体，包含策略网络、世界模型等组件
        env: 训练环境（通常是CARLA环境）
        replay: 经验回放缓冲区
        logger: 日志记录器
        args: 训练参数配置
    """
    # 创建日志目录
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    
    # 定义各种触发器，用于控制不同操作的频率
    should_expl = embodied.when.Until(args.expl_until)  # 探索阶段触发器，直到指定步数
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)  # 训练频率触发器
    should_log = embodied.when.Clock(args.log_every)  # 日志记录频率触发器
    should_save = embodied.when.Clock(args.save_every)  # 模型保存频率触发器
    should_sync = embodied.when.Every(args.sync_every)  # 同步频率触发器
    
    # 获取步数计数器和更新计数器
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    
    # 打印观测空间和动作空间信息
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    # 创建计时器，用于测量各个组件的运行时间
    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    # 用于记录出现非零值的键
    nonzeros = set()

    def per_episode(ep):
        """
        每个episode结束时的回调函数
        计算并记录episode的统计信息
        """
        # 计算episode长度和总奖励
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        
        # 记录episode级别的指标
        logger.add(
            {
                "length": length,  # episode长度
                "score": score,    # 总奖励
                "sum_abs_reward": sum_abs_reward,  # 绝对奖励和
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),  # 奖励率
            },
            prefix="episode",  # 添加前缀以区分不同类型的指标
        )
        print(f"Episode has {length} steps and return {score:.1f}.")
        
        # 准备要记录的统计信息
        stats = {}
        # 添加视频相关的键
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        
        # 根据正则表达式模式记录不同类型的统计信息
        for key, value in ep.items():
            # 如果配置为不记录零值，且当前键从未出现非零值，则跳过
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)  # 标记此键出现过非零值
            
            # 根据正则表达式匹配记录不同类型的统计值
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        
        # 将统计信息添加到metrics中
        metrics.add(stats, prefix="stats")

    # 创建驱动器，用于在环境中执行策略
    driver = embodied.Driver(env)
    # 注册episode结束时的回调函数
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep))
    # 注册每步的回调函数，用于增加步数计数器
    driver.on_step(lambda _, __, ___: step.increment())
    # 注册每步的回调函数，用于将转换添加到回放缓冲区
    driver.on_step(lambda tran, _, worker: replay.add(tran, worker))

    print("Prefill train dataset.")
    # 在开始正式训练前，先用随机策略填充回放缓冲区
    random_agent = embodied.RandomAgent(env.act_space, args.actor_dist_disc)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    
    # 记录初始指标并写入日志
    logger.add(metrics.result())
    logger.write()

    # 从回放缓冲区创建训练数据集
    dataset = agent.dataset(replay.dataset)
    # 用于存储训练状态的列表（使用列表是为了在内部函数中可修改）
    state = [None]
    batch = [None]

    def train_step(tran, worker, clock):
        """
        每个环境步骤的训练函数
        这是实际执行模型训练的地方
        """
        # 根据训练频率决定是否执行训练步骤
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                # 这里的 timer.scope("dataset") 是一个上下文管理器（context manager），它的作用是：

                # 测量时间：记录在 next(dataset) 操作上花费的时间
                # 分类统计：将这部分时间归类为 "dataset" 类型，用于性能分析
                # 从数据集中获取一个批次的数据
                batch[0] = next(dataset)
            
            # 执行智能体训练，返回输出、新状态和指标
            outs, state[0], mets = agent.train(batch[0], state[0])
            # 将训练指标添加到metrics中
            metrics.add(mets, prefix="train")

            # 如果回放缓冲区支持访问计数更新，则更新访问计数
            if getattr(replay, "update_visit_count", False):
                replay.update_visit_count(jax.device_get(batch[0]["env_step"]))

            # 如果输出中包含优先级信息，则更新回放缓冲区的优先级
            if "key" in outs:
                replay.prioritize(outs["key"], outs["env_step"], outs["model_loss"], outs["td_error"])

            # 增加更新计数器
            updates.increment()
        
        # 根据同步频率同步智能体（如果适用）
        if should_sync(updates):
            agent.sync()
        
        # 根据日志频率记录训练指标
        if should_log(step):
            # 获取聚合的指标
            agg = metrics.result()
            # 获取智能体的报告（例如，内部值的可视化）
            report = agent.report(batch[0])
            # 避免重复记录相同的键
            report = {k: v for k, v in report.items() if "train/" + k not in agg}
            
            # 添加各种指标到日志
            logger.add(agg)
            logger.add(report, prefix="report")
            logger.add(replay.stats, prefix="replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    # 注册每步的训练函数
    driver.on_step(train_step)

    # 创建检查点，用于保存和加载模型
    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step  # 将步数计数器添加到检查点
    checkpoint.agent = agent  # 将智能体添加到检查点
    checkpoint.replay = replay  # 将回放缓冲区添加到检查点
    
    # 如果指定了从检查点加载，则加载指定的检查点
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    
    # 加载现有检查点或保存初始检查点
    checkpoint.load_or_save()
    should_save(step)  # 标记刚刚已保存

    print("Start training loop.")
    # 开始训练循环
    driver._state = None
    # 定义策略函数，根据探索触发器决定使用探索模式还是训练模式
    policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")
    
    # 主训练循环：持续与环境交互直到达到最大步数
    while step < args.steps:
        driver(policy, steps=100)  # 每次执行100步
        # 如果到达保存频率，则保存检查点
        if should_save(step):
            checkpoint.save()
    
    # 最后写入日志
    logger.write()