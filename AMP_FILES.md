# AMP算法使用的文件清单

## 1. 核心配置文件

### 1.1 算法配置
- **`source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`**
  - 定义 `AmpAlgoCfg`：AMP算法配置（判别器参数、奖励系数等）
  - 定义 `G1ParkourPPORunnerCfg`：训练运行器配置

### 1.2 环境配置
- **`source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py`**
  - 定义 `G1ParkourEnvCfg`：完整的AMP环境配置
  - 配置 Motion Reference Manager
  - 配置 AMASS 运动数据路径

- **`source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`**
  - 定义 `ParkourEnvCfg`：基础环境配置
  - 定义 `AmpPolicyStateObsCfg`：策略状态观察配置
  - 定义 `AmpReferenceStateObsCfg`：参考运动状态观察配置
  - 定义场景、奖励、终止条件等

### 1.3 任务注册
- **`source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py`**
  - 注册任务：`Instinct-Parkour-Target-Amp-G1-v0`
  - 注册任务：`Instinct-Parkour-Target-Amp-G1-Play-v0`

## 2. Motion Reference 相关文件

### 2.1 核心管理器
- **`source/instinctlab/instinctlab/motion_reference/motion_reference_manager.py`**
  - `MotionReferenceManager`：管理参考运动数据
  - 提供参考帧数据给判别器

- **`source/instinctlab/instinctlab/motion_reference/motion_reference_cfg.py`**
  - `MotionReferenceManagerCfg`：Motion Reference配置类

- **`source/instinctlab/instinctlab/motion_reference/motion_reference_data.py`**
  - `MotionReferenceData`、`MotionReferenceState`：数据结构

- **`source/instinctlab/instinctlab/motion_reference/motion_buffer.py`**
  - 运动缓冲区管理

- **`source/instinctlab/instinctlab/motion_reference/__init__.py`**
  - 模块导出和AMP使用说明

### 2.2 运动文件处理
- **`source/instinctlab/instinctlab/motion_reference/motion_files/amass_motion.py`**
  - `AmassMotion`：从AMASS数据集加载运动数据

- **`source/instinctlab/instinctlab/motion_reference/motion_files/amass_motion_cfg.py`**
  - `AmassMotionCfg`：AMASS运动配置

- **`source/instinctlab/instinctlab/motion_reference/utils.py`**
  - 运动插值等工具函数（如 `motion_interpolate_bilinear`）

## 3. 观察空间相关文件

### 3.1 AMP专用观察函数
- **`source/instinctlab/instinctlab/envs/mdp/observations/reference_as_state.py`**
  - `projected_gravity_reference_as_state`：参考运动的投影重力
  - `joint_pos_rel_reference_as_state`：参考运动的相对关节位置
  - `joint_vel_rel_reference_as_state`：参考运动的相对关节速度
  - `base_lin_vel_reference_as_state`：参考运动的基座线速度
  - `base_ang_vel_reference_as_state`：参考运动的基座角速度
  - 其他参考状态观察函数

- **`source/instinctlab/instinctlab/envs/mdp/observations/motion_reference.py`**
  - 其他motion reference相关的观察函数

- **`source/instinctlab/instinctlab/envs/mdp/observations/reference_masked_proprioception.py`**
  - 带掩码的本体感觉观察函数

### 3.2 标准观察函数（用于策略状态）
- **`source/instinctlab/instinctlab/envs/mdp/observations/body.py`**
  - `projected_gravity`、`base_lin_vel`、`base_ang_vel` 等标准观察函数

- **`source/instinctlab/instinctlab/tasks/parkour/mdp/__init__.py`**
  - 导入所有MDP相关函数（包括观察、奖励、命令等）

## 4. MDP相关文件（Parkour任务）

### 4.1 奖励函数
- **`source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py`**
  - Parkour任务特定的奖励函数

### 4.2 命令生成
- **`source/instinctlab/instinctlab/tasks/parkour/mdp/commands/pose_velocity_command.py`**
  - `PoseVelocityCommandCfg`：速度命令生成

- **`source/instinctlab/instinctlab/tasks/parkour/mdp/commands/commands_cfg.py`**
  - 命令配置

### 4.3 其他MDP组件
- **`source/instinctlab/instinctlab/tasks/parkour/mdp/events.py`**
  - 环境事件处理

- **`source/instinctlab/instinctlab/tasks/parkour/mdp/curriculums.py`**
  - 课程学习

- **`source/instinctlab/instinctlab/tasks/parkour/mdp/terminations.py`**
  - 终止条件

## 5. 通用MDP文件

### 5.1 Motion Reference奖励
- **`source/instinctlab/instinctlab/envs/mdp/rewards/motion_reference.py`**
  - 各种motion reference相关的奖励函数（虽然AMP主要用判别器奖励）

### 5.2 观察函数
- **`source/instinctlab/instinctlab/envs/mdp/observations/__init__.py`**
  - 观察函数模块导出

## 6. 算法实现文件（在instinct_rl仓库中）

**注意**：WasabiPPO算法的实际实现在 `instinct_rl` 仓库中，不在本仓库。本仓库只提供配置。

根据配置，算法实现应该包含：
- WasabiPPO类（继承自PPO）
- 判别器网络实现
- 判别器训练逻辑
- 判别器奖励计算

## 7. 训练和测试脚本

- **`scripts/instinct_rl/train.py`**
  - 训练脚本（支持AMP任务）

- **`source/instinctlab/instinctlab/tasks/parkour/scripts/play.py`**
  - Parkour任务的测试/可视化脚本

- **`scripts/instinct_rl/play.py`**
  - 通用测试脚本

## 8. 工具和辅助文件

- **`source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py`**
  - `InstinctRlPpoAlgorithmCfg`：PPO算法基类配置
  - `InstinctRlOnPolicyRunnerCfg`：运行器基类配置

- **`source/instinctlab/instinctlab/assets/unitree_g1.py`**
  - G1机器人配置（关节映射、执行器等）

- **`source/instinctlab/instinctlab/tasks/parkour/README.md`**
  - Parkour任务使用说明

## 9. 文档文件

- **`DOCS.md`**
  - Motion Reference系统文档

- **`source/instinctlab/instinctlab/motion_reference/__init__.py`**
  - AMP使用FAQ

## 文件依赖关系图

```
g1_parkour_target_amp_cfg.py (环境配置)
    ├── parkour_env_cfg.py (基础环境配置)
    │   ├── observations (观察配置)
    │   │   ├── AmpPolicyStateObsCfg (策略状态)
    │   │   └── AmpReferenceStateObsCfg (参考状态)
    │   └── motion_reference_cfg (Motion Reference配置)
    │
    └── instinct_rl_amp_cfg.py (算法配置)
        └── WasabiPPO (在instinct_rl仓库中实现)

观察函数调用链：
    parkour_env_cfg.py
        ├── mdp.projected_gravity_reference_as_state
        │   └── reference_as_state.py
        ├── mdp.joint_pos_rel_reference_as_state
        │   └── reference_as_state.py
        └── ... (其他参考状态函数)

Motion Reference数据流：
    amass_motion.py (加载数据)
        └── motion_reference_manager.py (管理数据)
            └── reference_as_state.py (提取状态)
                └── WasabiPPO判别器 (训练)
```

## 关键文件说明

1. **配置入口**：`g1_parkour_target_amp_cfg.py` - 这是配置AMP任务的起点
2. **算法配置**：`instinct_rl_amp_cfg.py` - 定义判别器参数
3. **观察配置**：`parkour_env_cfg.py` 中的 `AmpPolicyStateObsCfg` 和 `AmpReferenceStateObsCfg`
4. **参考状态提取**：`reference_as_state.py` - 从motion reference提取状态
5. **Motion Reference管理**：`motion_reference_manager.py` - 核心数据管理
