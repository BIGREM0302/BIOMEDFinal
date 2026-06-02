import tf_keras as keras
import numpy as np

print("載入模型中...")
model = keras.models.load_model("monted")

# 直接獲取模型中所有含有權重的陣列 (W, b)
# 順序為：Conv1(W,b), Conv2(W,b), Dense1(W,b), Dense2(W,b), Dense3(W,b)
weights = model.get_weights()

# 將這 10 個 numpy 陣列存入壓縮檔
# 儲存名稱為 arr_0 到 arr_9
np.savez('weights.npz', *weights)
print("萃取成功！已生成 weights.npz")