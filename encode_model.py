"""

This file is used to convert .zip file, the trained model data, to .txt file,
which should be inserted into 'submission.py' for Kaggle submission.

"""

import base64

# 讀取你訓練好的模型檔案
with open("bci_connect4_cnn_ai.zip", "rb") as f:
    encoded_string = base64.b64encode(f.read()).decode("utf-8")

# 把這串文字存成一個文字檔
with open("model_string.txt", "w") as f:
    f.write(encoded_string)
print("? 模型已轉換成純文字！")