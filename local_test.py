"""

This script is designed for local testing of your ConnectX agent before you submit it to Kaggle.

"""

from kaggle_environments import evaluate, make
import numpy as np

# 💡 直接從你的提交檔載入最終的 Agent
# 確保你的 submission.py 已經填入了最新轉好的 Base64 模型字串
from submission import my_agent 

def run_matches(agent1, agent2, num_episodes=100):
    print(f"啟動 {num_episodes} 局對戰: {agent1} VS {agent2}")
    
    # 使用 Kaggle 官方的評估函數
    # 參數設定為標準四子棋 (6x7盤面, 連4獲勝)
    rewards = evaluate(
        "connectx", 
        [agent1, agent2], 
        num_episodes=num_episodes, 
        configuration={"rows": 6, "columns": 7, "inarow": 4}
    )
    
    # 統計勝負
    agent1_wins = sum(1 for r in rewards if r[0] == 1)
    agent2_wins = sum(1 for r in rewards if r[1] == 1)
    draws = sum(1 for r in rewards if r[0] == 0) # 雙方都是 0 算平局
    
    print(f"📊 戰績結算:")
    print(f"  - 玩家 1 ({agent1}) 勝率: {agent1_wins / num_episodes * 100:.1f}%")
    print(f"  - 玩家 2 ({agent2}) 勝率: {agent2_wins / num_episodes * 100:.1f}%")
    print(f"  - 平局率: {draws / num_episodes * 100:.1f}%\n")

if __name__ == "__main__":
    print("=== Kaggle 提交前本地端終極測試 ===")
    
    # 測試 1：虐菜測試 (確認沒有低級失誤)
    run_matches(my_agent, "random", num_episodes=50)
    
    # 測試 2：防守測試 (對抗 Kaggle 內建高手)
    # 注意：Negamax 當後手時非常難纏，能打平就算及格
    run_matches(my_agent, "negamax", num_episodes=20)
    
    # 測試 3：自我對弈測試 (矛盾大對決)
    # 觀察先手與後手的勝率差異
    run_matches(my_agent, my_agent, num_episodes=20)