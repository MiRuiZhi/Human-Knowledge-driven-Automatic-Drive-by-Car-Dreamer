# CarDreamer 项目 - 训练与验证命令行指南

## 一、环境准备

1. 安装 CARLA 模拟器 (0.9.15 版本)
2. 设置环境变量：
   ```bash
   export CARLA_ROOT="/path/to/carla"
   export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
   ```
3. 安装项目依赖：
   ```bash
   cd CarDreamer
   pip install flit
   flit install --symlink
   ```

## 二、模型训练

### 训练脚本：`train_dm3.sh`

**基本语法：**
```bash
bash train_dm3.sh <carla_port> <gpu_device> [additional_training_parameters]
```

**参数说明：**
- `<carla_port>`: CARLA 服务器使用的端口号
- `<gpu_device>`: 使用的 GPU 设备编号 (从0开始)
- `[additional_training_parameters]`: 其他训练参数

### 常用训练示例：

1. **基础训练命令（四车道任务）：**
   ```bash
   bash train_dm3.sh 2000 0 --task carla_lane_merge --dreamerv3.logdir ./logdir/carla_lane_merge
   ```

2. **覆盖特定参数的训练：**
   ```bash
   bash train_dm3.sh 2000 0 --task carla_lane_merge \
       --dreamerv3.logdir ./logdir/carla_lane_merge \
       --dreamerv3.run.steps=5e6
   ```

3. **使用不同模型大小进行训练：**
   ```bash
   bash train_dm3.sh 2000 0 --task carla_navigation --dreamerv3.logdir ./logdir/carla_navigation --dreamerv3=small
   ```

4. **更改训练步数：**
   ```bash
   bash train_dm3.sh 2000 0 --task carla_overtake \
       --dreamerv3.logdir ./logdir/carla_overtake \
       --dreamerv3.run.steps=1e7
   ```

## 三、可用任务列表

CarDreamer 提供了多种驾驶任务：

- `carla_navigation`: 导航任务
- `carla_four_lane`: 四车道任务  
- `carla_overtake`: 超车任务
- `carla_right_turn`: 右转任务
- `carla_left_turn`: 左转任务
- `carla_roundabout`: 环岛任务
- `carla_traffic_lights`: 红绿灯任务
- `carla_stop_sign`: 停车标志任务
- `carla_lane_merge`: 车道合并任务
- `carla_wpt`: 路点跟随任务
- `carla_wpt_fixed`: 固定路点跟随任务
- `carla_right_turn_random`: 随机右转任务

## 四、训练参数详解

1. **模型大小设置：**
   ```
   --dreamerv3=xsmall/small/medium/large/xlarge
   ```

2. **训练步数：**
   ```
   --dreamerv3.run.steps=<步数> (如 5e6 表示 500 万步)
   ```

3. **日志目录：**
   ```
   --dreamerv3.logdir <路径> (指定日志和检查点保存路径)
   ```

4. **批处理大小：**
   ```
   --dreamerv3.batch_size <大小>
   ```

5. **CARLA 端口：**
   ```
   --env.world.carla_port <端口号>
   ```

## 五、模型验证

### 验证脚本：`eval_dm3.sh`

**基本语法：**
```bash
bash eval_dm3.sh <carla_port> <gpu_device> <checkpoint_path> [additional_parameters]
```

**参数说明：**
- `<carla_port>`: CARLA 服务器使用的端口号
- `<gpu_device>`: 使用的 GPU 设备编号
- `<checkpoint_path>`: 检查点文件路径
- `[additional_parameters]`: 其他验证参数

### 验证示例：
```bash
bash eval_dm3.sh 2000 0 ./logdir/carla_four_lane/checkpoint.ckpt --task carla_four_lane --dreamerv3.logdir ./logdir/eval_carla_four_lane
```

### 分析验证结果：
```bash
python dreamerv3/eval_stats.py --logdir ./logdir/eval_carla_four_lane
```

## 六、可视化监控

1. **在线监控（训练过程中的实时数据显示）：**
   访问 `http://localhost:9000/` (端口号为 CARLA 端口号 + 7000)

2. **离线监控（使用 TensorBoard）：**
   ```bash
   tensorboard --logdir ./logdir/carla_four_lane
   ```
   然后在浏览器中打开 `http://localhost:6006/`

3. **使用 WandB 进行可视化（需要额外配置）：**
   在 `dreamerv3/train.py` 中添加 WandB logger

## 七、注意事项

1. 脚本会自动启动 CARLA 服务器，无需手动启动
2. 确保指定的 GPU 设备可用
3. 为避免端口冲突，可以选择不同的 CARLA 端口号
4. 训练日志会保存到指定的日志目录中，便于后续分析
5. 在最新实现中，成功率通过 `destination_reached` 指标计算，可能与论文略有差异
6. 使用 `checkpoint.ckpt` 文件进行模型评估
7. 训练过程中脚本会自动重启崩溃的进程，确保训练连续性