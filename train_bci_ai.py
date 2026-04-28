"""
This Python file will produce a file named "bci_connect4_cnn_ai.zip" in the current directory, 
which contains the trained model for the Connect Four game.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kaggle_environments import make
from stable_baselines3 import PPO
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv 
from stable_baselines3.common.monitor import Monitor
import random


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 🚀 殘差連接：讓神經網路可以訓得更深而不失真
        return torch.relu(out)
    
# ==========================================
# 🚀 階段零：打造專屬四子棋的 CNN 大腦
# ==========================================
class ConnectFourCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        # 使用殘差網路加深思考層次
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class ConnectFourEnv(gym.Env):
    def __init__(self):
        super(ConnectFourEnv, self).__init__()
        self.env = make("connectx", debug=False)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=2, shape=(1, 6, 7), dtype=np.float32)
        
        self.past_model = None # 用來存放「過去的自己」
        self.is_flipped = False # 紀錄這局是否被翻轉
        
        # 預設對手
        self.change_trainer("random")

    # 🚀 把被遺忘的 change_trainer 補回來！
    def change_trainer(self, opponent):
        print(f"🔄 環境設定：對手已更換為 {opponent}")
        self.trainer = self.env.train([None, opponent])


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 確保有基礎對手
        if not hasattr(self, 'trainer'):
            self.change_trainer("random")
            
        obs = self.trainer.reset()
        
        # 方案一：資料增強 (50% 機率翻轉棋盤)
        self.is_flipped = random.choice([True, False])
        board = np.array(obs.board).reshape(1, 6, 7)
        if self.is_flipped:
            board = np.flip(board, axis=2) # 左右翻轉
            
        return board.astype(np.float32), {}

    # 💡 1. 新增：把過去的模型包裝成 Kaggle 標準的 Agent 函數
    def past_model_agent(self, obs, config):
        if self.past_model is None:
            # 如果還沒有過去的模型，就隨機下
            return random.choice([c for c in range(7) if obs.board[c] == 0])
        
        # Kaggle 傳入的是一維陣列，轉回 (1, 6, 7)
        board = np.array(obs.board).reshape(1, 6, 7)
        
        # ⚠️ 視角欺騙 (非常重要)：
        # 模型訓練時習慣自己是「1(先手)」，但現在它是對手「2(後手)」
        # 我們必須把盤面上的 1 和 2 對調，騙過模型的大腦，它才會正確防守
        if obs.mark == 2:
            board_swapped = np.copy(board)
            board_swapped[board == 1] = 2
            board_swapped[board == 2] = 1
            board = board_swapped
            
        action, _ = self.past_model.predict(board.astype(np.float32), deterministic=True)
        return int(action)

    # 💡 2. 修改：把自製的 agent 餵給 Kaggle trainer
    def set_self_play_opponent(self, model_path):
        self.past_model = PPO.load(model_path, device="cpu")
        # 直接讓 Kaggle 引擎把我們的 past_model_agent 當作對手！
        self.trainer = self.env.train([None, self.past_model_agent])
        print("🤖 已載入過去的自己作為對手！")

    # 💡 3. 修改：乾淨俐落的 step 函數
    def step(self, action):
        actual_action = int(6 - action) if self.is_flipped else int(action)
        
        board_state = self.env.state[0].observation.board
        if board_state[int(actual_action)] != 0:
            return np.array(board_state).reshape(1, 6, 7).astype(np.float32), -10.0, True, False, {}

        # 現在我們只需要管好自己，呼叫 trainer.step
        # Kaggle 會自動在背景呼叫 past_model_agent 幫對手下棋！
        obs, reward, done, info = self.trainer.step(actual_action)
        board = np.array(obs.board).reshape(1, 6, 7)
            
        if self.is_flipped:
            board = np.flip(board, axis=2)
            
        reward = float(reward) if reward is not None else 0.0
        return board.astype(np.float32), reward, done, False, {}

# 💡 確保你的 make_env 長這樣 (有包上 Monitor 才能看到勝率)
def make_env(rank, opponent="negamax"):
    def _init():
        env = ConnectFourEnv()
        env.change_trainer(opponent)
        env = Monitor(env) # 加上監視器紀錄數據
        return env
    return _init

# ==========================================
# 🚀 階段二：執行兩階段課程訓練
# ==========================================
# ... (前面的 CNN 和 Env 類別定義維持不變) ...

if __name__ == "__main__":
    print(f"🔥 GPU 運算狀態: {torch.cuda.is_available()}")

    num_cpu = 4
    print(f"🧬 正在開啟 {num_cpu} 個並行環境")
    env = SubprocVecEnv([make_env(i, "random") for i in range(num_cpu)])

    policy_kwargs = dict(
        features_extractor_class=ConnectFourCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # ==========================================================
    # 設定：是否重新訓練 Stage 1
    # ==========================================================
    retrain_stage1 = False

    if retrain_stage1:
        print("\n[Stage 1] 從頭開始訓練，對手設定為 random...")
        env.env_method("change_trainer", "random")

        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            ent_coef=0.01,
            batch_size=64,
            verbose=1,
            device="cuda",
            policy_kwargs=policy_kwargs,
        )

        model.learn(total_timesteps=100000, reset_num_timesteps=True)
        model.save("bci_model_stage1.zip")
        print("✅ Stage 1 訓練完成，已儲存為 bci_model_stage1.zip")

    else:
        print("🧠 載入既有 Stage 1 模型...")
        model = PPO.load("bci_model_stage1.zip", env=env, device="cuda")

    # ==========================================================
    # Stage 2：對手切到 negamax，繼續訓練
    # ==========================================================
    print("\n[Stage 2] 對手切換為 negamax，開始進階訓練...")
    env.env_method("change_trainer", "negamax")

    for generation in range(5):
        print(f"\n[第 {generation} 代] 開始自我對弈進化...")
        model.learn(total_timesteps=100000, reset_num_timesteps=False)

        current_model_path = f"model_gen_{generation}.zip"
        model.save(current_model_path)

        env.env_method("set_self_play_opponent", current_model_path)
        print("🔄 對手已切換為：上一代的自己")

    model.save("bci_connect4_cnn_ai.zip")
    print("✅ Stage 2 訓練完成！最終模型已儲存為 bci_connect4_cnn_ai.zip")

    # ==========================================
    # 測試
    # ==========================================
    print("\n--- 模擬預測測試 ---")
    current_board = np.zeros((1, 1, 6, 7), dtype=np.float32)
    current_board[0, 0, 5, 3] = 1

    action, _ = model.predict(current_board, deterministic=True)
    best_col = int(action[0]) if isinstance(action, (np.ndarray, list)) else int(action)
    print(f"🎯 預測完成！建議落子第 {best_col + 1} 欄")

    env.close()