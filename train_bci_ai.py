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

# ==========================================
# 🚀 階段零：打造專屬四子棋的「微型 CNN 大腦」
# 解決預設 CNN (NatureCNN) 無法處理 6x7 小矩陣的問題
# ==========================================
class ConnectFourCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # 建立專為 6x7 設計的微型卷積神經網路
        # 使用 3x3 的掃描濾鏡 (Kernel size=3)，剛好可以看懂「連線」
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 讓 PyTorch 自動計算 CNN 攤平後的大小，以串接後面的線性層
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ==========================================
# 階段一：建立溝通橋樑 (Environment Wrapper)
# ==========================================
class ConnectFourEnv(gym.Env):
    def __init__(self):
        super(ConnectFourEnv, self).__init__()
        self.env = make("connectx", debug=False)
        self.trainer = self.env.train([None, "random"]) 
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=2, shape=(1, 6, 7), dtype=np.float32)

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


# ==========================================
# 階段二：訓練客製化 CNN 模型
# ==========================================
if __name__ == "__main__":
    print(f"🔥 GPU 是否可用: {torch.cuda.is_available()}")
    env = ConnectFourEnv()

    # 🚀 將我們剛才寫的 Custom CNN 餵給模型
    policy_kwargs = dict(
        features_extractor_class=ConnectFourCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # 建立模型，大膽指定使用 cuda！
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        device="cuda", 
        policy_kwargs=policy_kwargs
    )

    print("🧠 AI (專屬 CNN 架構) 開始使用 GPU 進行自我對弈訓練...")
    # 可以先用 50000 步測試，確定沒問題後睡前再改成 1000000 步
    model.learn(total_timesteps=50000)

    model.save("bci_connect4_cnn_ai")
    print("✅ 訓練完成！模型已儲存")


    # ==========================================
    # 階段三：實機預測測試
    # ==========================================
    print("\n--- 模擬 BCI 實機運作 ---")
    # 如果你要在沒有 GPU 的 Arduino 電腦上載入，記得把 "cuda" 改回 "cpu"
    loaded_model = PPO.load("bci_connect4_cnn_ai", device="cuda")

    # 🚀 修正 1：增加一個維度，變成 [Batch=1, Channel=1, Height=6, Width=7]
    # 這就相當於告訴模型：「我要送給你 1 張 6x7 的單色圖片」
    current_board = np.zeros((1, 1, 6, 7), dtype=np.float32)
    current_board[0, 0, 5, 3] = 1 # 假裝底層第4格有子

    # 取得 AI 建議的下一步
    action, _states = loaded_model.predict(current_board, deterministic=True)
    best_column = int(action[0]) # 🚀 修正 2：從 array 中取出第一個元素

    obs_tensor = torch.tensor(current_board).to("cuda")
    distribution = loaded_model.policy.get_distribution(obs_tensor)
    action_probs = distribution.distribution.probs.detach().cpu().numpy()[0]
    win_confidence = action_probs[best_column] * 100 

    print(f"🎯 預測完成！建議落子第 {best_column + 1} 欄，信心水準: {win_confidence:.1f} %")