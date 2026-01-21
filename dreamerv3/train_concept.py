import datetime
import warnings

# 导入embodied框架，这是DreamerV3的基础框架
import embodied
# 导入yaml解析库，用于读取配置文件
import ruamel.yaml as yaml

# 导入CarDreamer环境模块
import car_dreamer
# 导入DreamerV3算法实现
import dreamerv3

# 过滤掉一些警告信息，避免干扰训练过程的输出
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def wrap_env(env, config):
    """
    包装环境函数，对环境进行一系列标准化处理
    - 将离散动作转换为one-hot编码
    - 对连续动作进行归一化
    - 可选地对连续动作进行离散化
    - 添加时间限制、检查空间等
    """
    # 获取wrapper配置
    args = config.wrapper
    
    # 包装环境信息
    env = embodied.wrappers.InfoWrapper(env)
    
    # 遍历动作空间，对不同类型的动作进行处理
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            # 如果是离散动作，转换为one-hot编码
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            # 如果设置了离散化参数，对连续动作进行离散化
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            # 否则对连续动作进行归一化处理
            env = embodied.wrappers.NormalizeAction(env, name)
    
    # 展开标量维度，使所有观测具有一致的形状
    env = embodied.wrappers.ExpandScalars(env)
    
    # 如果设置了时间限制，则添加时间限制包装器
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    
    # 如果开启了检查模式，则添加空间检查包装器
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    
    # 再次遍历动作空间，对连续动作进行裁剪
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    
    return env


def main(argv=None):
    """
    主函数，执行训练流程
    包括：加载配置 -> 创建环境 -> 创建智能体 -> 开始训练
    """
    # 加载DreamerV3的模型配置文件
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    
    # 初始化配置，首先使用默认配置
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    # 然后更新为small模型配置（较小的模型用于测试或资源受限环境）
    config = config.update({"dreamerv3": model_configs["small"]})

    # 解析命令行参数，其中task参数默认为carla_navigation
    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    
    # 遍历所有指定的任务，创建环境和更新配置
    for name in parsed.task:
        print("Using task: ", name)
        # 创建Carla环境及对应的配置
        env, env_config = car_dreamer.create_task(name, argv)
        # 将环境配置合并到主配置中
        config = config.update(env_config)
    
    # 解析剩余的命令行参数
    config = embodied.Flags(config).parse(other)
    print(config)

    # 创建日志目录
    logdir = embodied.Path(config.dreamerv3.logdir)
    # 创建计数器，用于跟踪训练步数
    step = embodied.Counter()
    
    # 创建日志记录器，将指标输出到多个目标：
    # 1. 终端输出 - 实时查看训练进度
    # 2. JSONL文件 - 结构化存储用于后续分析
    # 3. TensorBoard - 可视化训练过程
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),  # 终端输出
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),  # JSONL格式输出
            embodied.logger.TensorBoardOutput(logdir),  # TensorBoard输出
        ],
    )

    # 导入gym环境包装器
    from embodied.envs import from_gym

    # 提取DreamerV3的配置
    dreamerv3_config = config.dreamerv3
    
    # 将CARLA环境包装为FromGym格式，以便与embodied框架兼容
    env = from_gym.FromGym(env)
    # 对环境进行标准化包装
    env = wrap_env(env, dreamerv3_config)
    # 将环境包装为批量环境，这里只使用一个环境实例
    env = embodied.BatchEnv([env], parallel=False)

    # 保存当前配置到日志目录，用于复现实验
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    print(f"[Train] Config saved to {logdir / config_filename}")

    # 创建DreamerV3智能体
    # 输入参数包括：观测空间、动作空间、步数计数器、配置
    agent = dreamerv3.ConceptAgent(env.obs_space, env.act_space, step, dreamerv3_config)
    
    # 创建均匀随机回放缓冲区，用于存储经验数据
    # 参数包括：序列长度、回放缓冲区大小、回放缓冲区保存路径
    replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")
    
    # 构建运行参数配置
    args = embodied.Config(
        **dreamerv3_config.run,  # 展开运行配置
        logdir=dreamerv3_config.logdir,  # 日志目录
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,  # 每个批次的总步数
        actor_dist_disc=dreamerv3_config.actor_dist_disc,  # 行为者分布离散化参数
    )
    
    # 启动训练流程，传入智能体、环境、回放缓冲区、日志记录器和参数
    # 这里调用了embodied框架提供的标准训练函数
    embodied.run.train(agent, env, replay, logger, args)


if __name__ == "__main__":
    # 当作为主程序运行时，调用main函数开始训练
    # 为了调试目的，可以在此处添加特定的参数
    import sys
    
    # 使用用户提供的参数作为预定义参数
    predefined_argv = [
        "--task", "carla_lane_merge",
        "--dreamerv3.logdir", "./logdir/carla_lane_merge",
        "--dreamerv3.run.steps", "5e6",
    ]
    
    # 如果命令行提供了参数，则使用命令行参数，否则使用预定义参数
    if len(sys.argv) > 1:
        main()
    else:
        print("使用预定义参数运行...")
        main(predefined_argv)