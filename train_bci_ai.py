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

# ==========================================
# 🚀 階段零：打造專屬四子棋的 CNN 大腦
# ==========================================
class ConnectFourCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# ==========================================
# 🚀 階段一：建立支援「切換對手」的環境
# ==========================================
class ConnectFourEnv(gym.Env):
    def __init__(self):
        super(ConnectFourEnv, self).__init__()
        self.env = make("connectx", debug=False)
        # 預設對手先設為 random
        self.change_trainer("random")
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=2, shape=(1, 6, 7), dtype=np.float32)

    # 💡 新增：隨時切換對手的函數
    def change_trainer(self, opponent):
        print(f"🔄 環境設定：對手已更換為 {opponent}")
        self.trainer = self.env.train([None, opponent])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.trainer.reset()
        return np.array(obs.board).reshape(1, 6, 7).astype(np.float32), {}

    def step(self, action):
        board = self.env.state[0].observation.board
        if board[int(action)] != 0:
            return np.array(board).reshape(1, 6, 7).astype(np.float32), -10.0, True, False, {}
        obs, reward, done, info = self.trainer.step(int(action))
        reward = float(reward) if reward is not None else 0.0
        return np.array(obs.board).reshape(1, 6, 7).astype(np.float32), reward, done, False, info

def make_env(rank, opponent="random"):
    def _init():
        env = ConnectFourEnv()
        env.change_trainer(opponent) # 設定該環境的對手
        env = Monitor(env) 
        return env
    return _init

# ==========================================
# 🚀 階段二：執行兩階段課程訓練
# ==========================================
# ... (前面的 CNN 和 Env 類別定義維持不變) ...

if __name__ == "__main__":
    print(f"🔥 GPU 運算狀態: {torch.cuda.is_available()}")
    
    # 🚀 設定並行環境 (對手設定為 negamax，因為我們要直接進入第二階段)
    num_cpu = 4 
    print(f"🧬 正在開啟 {num_cpu} 個並行環境 (Stage 2 模式: Negamax)")
    env = SubprocVecEnv([make_env(i, "negamax") for i in range(num_cpu)])

    policy_kwargs = dict(
        features_extractor_class=ConnectFourCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # ==========================================================
    # 模式 A：從頭開始訓練 (目前已註解)
    # ==========================================================
    """
    # 如果要重新從 Stage 1 開始，請取消下面這幾行的註解，並註解掉下方的模式 B
    env.env_method("change_trainer", "random") # 先換成 random 練基本功
    model = PPO("CnnPolicy", env, verbose=1, device="cuda", policy_kwargs=policy_kwargs)
    print("\n[Stage 1] 開始 10 萬步基礎訓練...")
    model.learn(total_timesteps=100000)
    model.save("bci_model_stage1")
    """

    # ==========================================================
    # 模式 B：直接載入 Stage 1 成果繼續練 (目前預設開啟)
    # ==========================================================
    print("🧠 正在載入 Stage 1 (bci_model_stage1.zip) 訓練成果...")
    # 注意：我們直接把 model 指向載入好的模型，並連接到新的並行環境
    model = PPO.load("bci_model_stage1", env=env, device="cuda")

    # --- 🎓 第二階段：研究所 (Negamax) ---
    print("\n[Stage 2] 開始 50 萬步進階訓練 (對手: Negamax)...")
    
    # 確保環境裡面的對手都是 negamax (雖然初始化已設，但保險起見再廣播一次)
    env.env_method("change_trainer", "negamax")
    
    # 繼續訓練
    # total_timesteps 設為 500,000，reset_num_timesteps=True 讓進度條從 0/500000 開始
    model.learn(total_timesteps=500000, reset_num_timesteps=True)
    
    # 🚀 存成最終專案名稱
    model.save("bci_connect4_cnn_ai")
    print("✅ Stage 2 訓練完成！最終模型已儲存為 bci_connect4_cnn_ai.zip")

    # ==========================================
    # 🚀 階段三：實機預測測試
    # ==========================================
    print("\n--- 模擬 BCI 實機運作測試 ---")
    current_board = np.zeros((1, 1, 6, 7), dtype=np.float32)
    current_board[0, 0, 5, 3] = 1 

    action, _ = model.predict(current_board, deterministic=True)
    best_col = action[0] if isinstance(action, (np.ndarray, list)) else action
    print(f"🎯 預測完成！建議落子第 {int(best_col) + 1} 欄")
    
    env.close()