"""Handover environment for training and testing the DRL agent."""

# 导入操作系统接口模块，用于路径操作
import os
# 导入类型提示相关模块：Any（任意类型）、Optional（可选类型）、TYPE_CHECKING（类型检查时导入）
from typing import Any, Optional, TYPE_CHECKING
# Gymnasium提供了一系列预定义的环境，并定义了一套统一的接口
# 导入 gymnasium 库，用于创建强化学习环境
import gymnasium as gym
# 从 gymnasium 中导入 spaces 模块，用于定义动作和观察空间
from gymnasium import spaces
# 导入 numpy 库，用于数值计算
import numpy as np

# 从当前包中导入 HOProcedurePPO 类，该类实现了切换过程的协议逻辑（状态机）
from .ho_protocol_ppo import HOProcedurePPO

# 如果是类型检查阶段，则导入 Config 和 stable_baselines3（避免循环导入）
if TYPE_CHECKING:
    from ..config import Config
    import stable_baselines3

# 基于 PyTorch 的强化学习算法库，提供了多种强化学习算法的实现
# 包括 PPO、A2C、DDPG、SAC 等

#使用Gymnasium和stable_baselines3,需要环境建模，算法部分由stable_baselines3实现

# 定义 HandoverEnvPPO 类，继承自 gym.Env，是一个自定义的 Gymnasium 环境
class HandoverEnvPPO(gym.Env):
    """
    Handover Environment for the PPO agent.

    This environment is used for training and testing the PPO agent.
    """

    # 类属性：当前文件的绝对路径
    path_to_env = os.path.abspath(__file__)
    # 类属性：环境 ID 计数器，用于给每个环境实例分配唯一 ID
    env_id: int = 0

    def __init__( # 构造函数，接收配置对象和三个数据列表（RSRP、SINR、归一化 SINR 的轨迹数据）
        self,
        config: "Config",
        rsrp_list: list[np.ndarray],
        sinr_list: list[np.ndarray],
        sinr_norm_list: list[np.ndarray],
    ):
        """Initialize the Handover Environment"""
        super().__init__() # 调用父类构造函数

        # Environment parameters
       # 环境参数
        HandoverEnvPPO.env_id += 1            # 环境 ID 自增
        self.config = config                   # 保存配置
        self.test_mode_on = False               # 测试模式标志，初始为 False
        self.terminate_on_rlf = config.terminate_on_rlf   # 是否因 RLF 终止 episode（从配置读取）
        self.terminate_on_pp = config.terminate_on_pp     # 是否因 Ping-Pong 终止 episode（从配置读取）

        # 训练时是否允许切换准备中止，初始为 False
        self.train_permit_ho_prep_abort = False
        # 奖励 shaping 系数，初始为 0.5（当前未在奖励中使用）
        self.shaping_scale = 0.5
        # 创建 HOProcedurePPO 实例，用于管理切换状态机
        self.ho_procedure = HOProcedurePPO(self.config)
        # 设置 ho_procedure 的 permit_ho_prep_abort 属性：测试模式下按配置，否则按训练标志（当前 False）
        self.ho_procedure.permit_ho_prep_abort = (
            self.config.permit_ho_prep_abort if self.test_mode_on else self.train_permit_ho_prep_abort
        )

        # Data loader
        # 数据加载器：保存传入的数据列表
        self.rsrp_list = rsrp_list
        self.sinr_list = sinr_list
        self.sinr_norm_list = sinr_norm_list

        # Environment parameters
        # 环境参数
        self.n_datasets = len(rsrp_list)                 # 数据集数量
        self.dataset_idx: int = 0                         # 当前使用的数据集索引，默认 0
        self.time_steps, self.n_bs = rsrp_list[0].shape   # 获取时间步数和基站数
        self.t = 0                                         # 当前时间步

        # 基站排列顺序，初始为 [0,1,2,...]
        self._perm = np.arange(self.n_bs)
        # 获取当前数据集的数据（根据 dataset_idx 和排列顺序）
        self._rsrp_ep = self.rsrp_list[self.dataset_idx]
        self._sinr_ep = self.sinr_list[self.dataset_idx]
        self._sinr_norm_ep = self.sinr_norm_list[self.dataset_idx]

        # Observation space
         # 观察空间的定义
        self.n_observations = 2 * self.n_bs + 1           # 观察向量维度：服务小区 one-hot (n_bs) + 归一化 SINR (n_bs) + PP 挂起标志 (1)
        self.o_low = np.zeros(self.n_observations, dtype=np.float32)   # 下界全 0
        self.o_high = np.ones(self.n_observations, dtype=np.float32)   # 上界全 1
        self.observation_space = spaces.Box(self.o_low, self.o_high, dtype=np.float32)  # 连续 Box 空间

        # Action space
        # 动作空间：离散，可选基站数量
        self.action_space = spaces.Discrete(self.n_bs)

        # Observations, flags, etc.
         # 以下为记录 episode 轨迹的列表，初始化为空
        self.s_action = []          # 动作序列

        # RSRP and SINR values
        self.s_rsrp = []
        self.s_sinr = []

        # Cell IDs
        self.s_pcell = []  # Serving cell
        self.s_tcell = []  # Target cell

        # Handover flags
        # 切换相关标志
        self.s_ho_complete = []      # 切换完成标志
        self.s_ho_prep = []          # 切换准备中标志
        self.s_ho_exec = []          # 切换执行中标志
        self.s_q_out_db = []         # 是否处于 out-of-sync 状态（Q_out）
        self.s_rlf = []              # RLF 发生标志
        self.s_pp = []               # Ping-Pong 发生标志

        # Relative value of counters (t/t_max)
        # 相对计数器值（当前值 / 最大允许值，范围 0~1）
        self.s_rel_n310_t310 = []    # N310/T310 计数器相对值
        self.s_rel_ho_prep_cnt = []  # 切换准备计数器相对值
        self.s_rel_ho_exec_cnt = []  # 切换执行计数器相对值
        self.s_rel_mtsc_cnt = []     # 移动性 T310 计数器相对值（用于 Ping-Pong 检测）

        # General state
        # 通用状态
        self.t = 0                    # 当前时间步
        self.state: np.ndarray | None = None   # 当前观察向量
        self.terminated = False       # 终止标志（因 RLF 或 PP）
        self.truncated = False        # 截断标志（达到最大时间步）

        # Reset environment
        # 重置环境
        self.reset()

    # 类方法：重置环境 ID 计数器为 0
    @classmethod
    def reset_cls(cls):
        """Reset the environment ID counter to 0."""
        cls.env_id = 0
    
    # 设置测试模式
    def set_test_mode(self, test_mode_on):
        """
        Set the test mode on or off: if on, the environment will not terminate on RLF or PP.

        Parameters
        ----------
        test_mode_on : bool
            Test mode flag.
        """
        self.test_mode_on = test_mode_on
        if self.test_mode_on:
             # 测试模式下：不因 RLF 或 PP 终止
            self.terminate_on_pp = False
            self.terminate_on_rlf = False
            # 允许切换准备中止（按配置）
            self.ho_procedure.permit_ho_prep_abort = self.config.permit_ho_prep_abort
        else:
            # 训练模式下：恢复配置中的终止条件
            self.terminate_on_pp = self.config.terminate_on_pp
            self.terminate_on_rlf = self.config.terminate_on_rlf
            # 禁止切换准备中止，以促进探索
            self.ho_procedure.permit_ho_prep_abort = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,  # 随机种子（未使用）
        options: Optional[dict[str, Any]] = None,  # 选项（未使用）
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed, by default None (not used).
        options : dict, optional
            Options for the reset, by default None (not used).

        Returns
        -------
        np.ndarray
            Initial observation.
        """
        # Observations, flags, etc.

       # 清空轨迹列表
        self.s_action = []

        # 调用父类的 reset 方法（处理 seed 等）
        super().reset()

        # 如果不是测试模式，随机选择一个数据集
        if not self.test_mode_on:
            self.dataset_idx = np.random.randint(0, self.n_datasets)
        # 获取当前数据集的形状（时间步数、基站数）
        self.time_steps, self.n_bs = self.rsrp_list[self.dataset_idx].shape
         # 根据测试模式决定是否打乱基站顺序（测试模式不打乱，训练模式随机打乱）
        if self.test_mode_on:
            self._perm = np.arange(self.n_bs)
        else:
            self._perm = np.random.permutation(self.n_bs)
        # 根据排列顺序获取当前 episode 的数据
        self._rsrp_ep = self.rsrp_list[self.dataset_idx][:, self._perm]
        self._sinr_ep = self.sinr_list[self.dataset_idx][:, self._perm]
        self._sinr_norm_ep = self.sinr_norm_list[self.dataset_idx][:, self._perm]

        # 重新初始化记录列表（确保清空）
        # RSRP and SINR values
        self.s_rsrp = []
        self.s_rsrp_unscaled = []
        self.s_rsrp_rel_to_pcell = []
        self.s_sinr = []
        self.s_sinr_unscaled = []

        # Cell IDs
        self.s_pcell = []
        self.s_tcell = []
        self.s_connected_idxs = []

        # Handover flags
        self.s_ho_complete = []
        self.s_ho_prep = []
        self.s_ho_exec = []
        self.s_q_out_db = []
        self.s_rlf = []
        self.s_pp = []

        # Relative value of counters (t/t_max)
        self.s_rel_n310_t310 = []
        self.s_rel_ho_prep_cnt = []
        self.s_rel_ho_exec_cnt = []
        self.s_rel_mtsc_cnt = []

        # General state
        # 重置时间步和终止标志
        self.t = 0
        self.terminated = False
        self.truncated = False

        # Reset the handover procedure
        # 重新创建 HOProcedurePPO 实例（重置状态机）
        self.ho_procedure = HOProcedurePPO(self.config)
        # 设置允许切换准备中止的标志
        self.ho_procedure.permit_ho_prep_abort = (
            self.config.permit_ho_prep_abort if self.test_mode_on else self.train_permit_ho_prep_abort
        )

        # Return the initial observation
        # 返回初始观察
        return self._get_initial_observation()

    # 私有方法：生成初始观察
    def _get_initial_observation(self) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Get the initial observation from the environment.

        Returns
        -------
        tuple[np.ndarray, dict]
            Initial observation and empty dictionary (info).
        """
       # 获取当前时间步的 RSRP
        rsrp = self._rsrp_ep[self.t, :]

        # Init cell = cell with highest RSRP
        # 初始服务小区选择 RSRP 最高的基站
        pcell = np.argmax(rsrp, axis=0)

        # Initialize the environment states
        # 记录初始动作（即服务小区）
        self.s_action.append(pcell)

        # 初始化各种标志为 False
        self.s_ho_complete.append(False)
        self.s_ho_prep.append(False)
        self.s_ho_exec.append(False)
        self.s_q_out_db.append(False)
        self.s_rlf.append(False)
        self.s_pp.append(False)

        # 初始化相对计数器为 0
        self.s_rel_n310_t310.append(0.0)
        self.s_rel_ho_prep_cnt.append(0.0)
        self.s_rel_ho_exec_cnt.append(0.0)
        self.s_rel_mtsc_cnt.append(0.0)

        # Initial state
        # 构建初始观察向量
        s_pcell_indicator = np.zeros(self.n_bs)      # 服务小区 one-hot
        s_pcell_indicator[pcell] = 1
        input_sinr = self._sinr_norm_ep[self.t, :]   # 当前归一化 SINR
        s_pp_indicator = np.array([0])                # PP 挂起标志，初始为 0
        self.state = np.concatenate((s_pcell_indicator, input_sinr, s_pp_indicator))

        # 返回观察和空 info 字典
        return np.array(self.state, dtype=np.float32), {}

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment"""
        # Validate the action and initial state
        # 如果 action 是 numpy 数组，转换为标量
        if isinstance(action, np.ndarray):
            action = action.item()
        # 验证动作和状态
        self._validate_state_action(action)

        # Take a step in the handover environment/state machine
        # 获取当前 RSRP 和 SINR
        rsrp = self._rsrp_ep[self.t, :]
        sinr = self._sinr_ep[self.t, :]
        # 调用 ho_procedure 的 step 方法，更新切换状态机，返回原始观察（包含各种标志）
        raw_obs = self.ho_procedure.step(rsrp, sinr, action)

        # Update the environment state
        # 更新环境内部记录（动作、标志等）
        self._update_state(action, raw_obs)

        # Get the reward
        reward = self._get_reward()

        # Terminate episode if RLF flag is set
        # First Phase & Second Phase: RLF terminates the episode
        # 如果配置了因 RLF 终止且 RLF 发生，则设置终止标志
        if self.terminate_on_rlf and raw_obs["rlf"]:
            self.terminated = True

        # Terminate episode if ping-pong flag is set
        # Second Phase only: PP also terminates the episode (controlled by self.terminate_on_pp)
         # 如果配置了因 Ping-Pong 终止且 PP 发生，则设置终止标志
        if self.terminate_on_pp and raw_obs["pp"]:
            self.terminated = True

        # Truncate episode if max episode length is reached
        # First Phase & Second Phase: Max steps reached
        # 如果达到最后一个时间步（即下一个时间步是终点），设置截断标志
        if 1 + self.t == self.time_steps - 1:
            self.truncated = True

        self.t += 1

        # 构建下一个观察向量
        s_bs = np.zeros(self.n_bs)                       # 服务小区 one-hot
        pcell = self.ho_procedure.rrc.pcell               # 获取当前服务小区（可能为 None）
        if pcell is not None:
            s_bs[pcell] = 1
        s_sinr = self._sinr_norm_ep[self.t, :]            # 下一个时间步的归一化 SINR
        s_pp = np.array([self.ho_procedure.cntr["mtsc"].pending])  # PP 计数器是否挂起
        self.state = np.concatenate((s_bs, s_sinr, s_pp))

        # 特殊处理：如果在训练且配置了跳过执行和 RLF 恢复，则自动快进时间步
        if (not self.test_mode_on) and self.config.skip_exec_and_rlfr_during_train:
            while ( # 当切换执行计数器或 RLF 恢复计数器挂起，且未到最后一个时间步时，循环快进
                (self.ho_procedure.cntr["ho_exec"].pending or self.ho_procedure.cntr["rlfr"].pending)
                and (1 + self.t < self.time_steps - 1)
            ):
                rsrp_ff = self._rsrp_ep[self.t, :]
                sinr_ff = self._sinr_ep[self.t, :]
                # 使用上一个动作继续步进（不产生新动作）
                raw_obs_ff = self.ho_procedure.step(rsrp_ff, sinr_ff, int(self.s_action[-1]))
                self._update_state(int(self.s_action[-1]), raw_obs_ff)
                self.t += 1

            # 快进结束后重新构建状态
            s_bs = np.zeros(self.n_bs)
            pcell = self.ho_procedure.rrc.pcell
            if pcell is not None:
                s_bs[pcell] = 1
            s_sinr = self._sinr_norm_ep[self.t, :]
            s_pp = np.array([self.ho_procedure.cntr["mtsc"].pending])
            self.state = np.concatenate((s_bs, s_sinr, s_pp))

        # 构造 info 字典，默认包含 sinr_lin 占位
        info = {"sinr_lin": 0}
        # 如果 episode 结束，添加统计信息
        if self.terminated or self.truncated:
            info["episode_metrics"] = self.get_statistics()

        # 返回步进结果（状态、奖励、终止标志、截断标志、info 字典）
        return (
            np.array(tuple(self.state), dtype=np.float32),
            reward,
            self.terminated,
            self.truncated,
            info,
        )

    # 验证动作和状态
    def _validate_state_action(self, action: int):
        """Validate the action and state before taking a step"""
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

    # 更新环境内部记录
    def _update_state(self, action: int, raw_obs: dict):
        """Update the environment state based on the action and various flags"""
        # 记录动作
        self.s_action.append(action)

        # 记录当前 RSRP 和 SINR
        rsrp = self._rsrp_ep[self.t, :]
        sinr = self._sinr_ep[self.t, :]
        self.s_rsrp.append(rsrp)
        self.s_sinr.append(sinr)

         # 获取当前服务小区和目标小区
        pcell = self.ho_procedure.get_pcell()
        tcell = self.ho_procedure.get_tcell()

        # 根据 pcell 是否为空，记录小区信息
        if pcell is None:
            self.s_pcell.append(-1)                         # 无服务小区
            if tcell is None:  # Radio link failure
                self.s_tcell.append(-1)                      # 无目标小区
                self.s_connected_idxs.append(np.zeros(self.n_bs))  # 无连接
            else:  # Handover execution
                self.s_tcell.append(int(tcell))
                connected_idxs = np.zeros(self.n_bs)
                connected_idxs[tcell] = 1
                self.s_connected_idxs.append(connected_idxs)  # 连接至目标小区
        else:
            self.s_pcell.append(int(pcell))
            connected_idxs = np.zeros(self.n_bs)
            connected_idxs[pcell] = 1
            self.s_connected_idxs.append(connected_idxs)      # 连接至服务小区
            if tcell is None:  # Normal operation
                self.s_tcell.append(-1)
            else:  # Handover preparation
                self.s_tcell.append(tcell)
        
        # 记录各种标志
        self.s_ho_complete.append(raw_obs["ho_complete"])
        self.s_ho_prep.append(raw_obs["ho_prep"])
        self.s_ho_exec.append(raw_obs["ho_exec"])
        self.s_q_out_db.append(raw_obs["q_in_db_out"])
        self.s_rlf.append(raw_obs["rlf"])
        self.s_pp.append(raw_obs["pp"])

        # 记录相对计数器
        self.s_rel_n310_t310.append(raw_obs["n310_t310_rel_cnt"])
        self.s_rel_ho_prep_cnt.append(raw_obs["ho_prep_rel_cnt"])
        self.s_rel_ho_exec_cnt.append(raw_obs["ho_exec_rel_cnt"])
        self.s_rel_mtsc_cnt.append(raw_obs["mtsc_rel_cnt"])

    # 返回所有记录列表的集合（用于调试）
    def _get_state_list(self):
        """Get the state list for the environment"""
        return [
            # Actions
            self.s_action,
            # RSRP and SINR values
            self.s_rsrp,
            self.s_sinr,
            # Cell IDs
            self.s_pcell,
            self.s_tcell,
            self.s_connected_idxs,
            # Handover flags
            self.s_ho_complete,
            self.s_ho_prep,
            self.s_ho_exec,
            self.s_q_out_db,
            self.s_rlf,
            self.s_pp,
            # Relative value of counters (t/t_max)
            self.s_rel_n310_t310,
            self.s_rel_ho_prep_cnt,
            self.s_rel_ho_exec_cnt,
            self.s_rel_mtsc_cnt,
        ]

    # 计算奖励
    def _get_reward(self) -> float:
        """Get the reward for the current state."""
        reward = 0.0

        # 复制当前 SINR，将 NaN 替换为 -inf（负无穷）
        sinr = self._sinr_ep[self.t, :].copy()
        sinr[np.isnan(sinr)] = -np.inf
        # 归一化 SINR，处理 NaN、inf 为 0 或 1
        sinr_norm = np.nan_to_num(
            self._sinr_norm_ep[self.t, :], nan=0.0, posinf=1.0, neginf=0.0
        )

        # 找出最佳基站（SINR 最大）
        best_bs = np.argmax(sinr)

        # r_SINR: use the chosen action; add +C only if action selects the current best BS
        a = int(self.s_action[-1])
        # 基础奖励：动作对应基站的归一化 SINR 值
        reward += sinr_norm[a].item()
        # 如果所选动作是最佳基站，额外加常数
        if a == best_bs:
            reward += self.config.rew_const  

        # r_PP: penalty if a ping-pong event occurs
        # r_PP：如果发生 Ping-Pong，减去常数
        if self.ho_procedure.pp_detected:
            reward -= self.config.rew_const #施加 -C 惩罚

        # r_RLF: penalty only when an RLF occurs
        # r_RLF：如果发生 RLF，减去常数
        # 检查 RLF 和失步状态，优先判断 RLF（最严重的惩罚）
        if self.s_rlf[-1]:                     # RLF 发生
            reward -= 2 * self.config.rew_const  # 惩罚 -2C
        elif self.s_q_out_db[-1]:               # 处于 out-of-sync（SINR < Q_out，但尚未触发 RLF）
            reward -= self.config.rew_const     # 惩罚 -C

        return reward

    # 设置数据集索引（用于测试特定数据集）
    def set_dataset_idx(self, dataset_idx):
        """Set the dataset index."""
        self.dataset_idx = dataset_idx
        self.time_steps, self.n_bs = self.rsrp_list[dataset_idx].shape
        self.reset()

    # 渲染方法（空实现，符合 Gymnasium 接口）
    def render(self, mode="human"):
        """Render the environment"""

    # 获取 episode 统计信息（通过 ho_procedure）
    def get_statistics(self):
        """Get statistics of the environment."""
        return self.ho_procedure.get_statistics(self._sinr_ep)

    # 设置训练阶段参数
    def set_training_phase(self, permit_abort: bool, shaping_scale: float | None = None):
        self.train_permit_ho_prep_abort = permit_abort
        if shaping_scale is not None:
            self.shaping_scale = max(0.0, float(shaping_scale))
        if not self.test_mode_on:
            self.ho_procedure.permit_ho_prep_abort = self.train_permit_ho_prep_abort

    # 属性方法：返回 episode 是否结束（终止或截断）
    @property
    def done(self):
        """Check if the episode is done (terminated or truncated)."""
        return self.terminated or self.truncated

# 外部测试函数：在给定环境上运行 PPO 模型，返回总奖励
def test_ppo_model(
    env: HandoverEnvPPO, model: "stable_baselines3.PPO", dataset_idx: int
) -> int:
    """
    Test the PPO model on the environment.

    Parameters
    ----------
    env : HandoverEnvPPO
        Handover environment.
    model : stable_baselines3.PPO
        PPO model.
    dataset_idx : int
        Dataset index.

    Returns
    -------
    int
        Total episode reward.
    """

    # Set the environment to test mode
    # 设置测试模式
    env.set_test_mode(True)
    # 设置要测试的数据集
    env.set_dataset_idx(dataset_idx)

    # Reset the environment and get the initial observation
    obs, _ = env.reset()

    truncated = False
    reward_arr = []
    while not truncated:
        # Predict the action
        action, _ = model.predict(
            obs, deterministic=env.config.test_deterministic_actions
        )

        # Take the action in the environment
        obs, reward, _, truncated, _ = env.step(action)

        # Store results
        reward_arr.append(reward)

    return np.sum(reward_arr)
