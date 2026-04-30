import pandas as pd
import ast
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 💡 核心修復：統一全部從 tf_keras 引入，杜絕混血衝突！
import tf_keras as keras  
from tf_keras import layers, models 

# ==========================================
# 1. 讀取資料
# ==========================================
df = pd.read_csv('final.csv').dropna()

dataList = []
resultList = []

print('\n📥 開始解析盤面特徵 (這個步驟最耗時，請稍候)...')
for i in tqdm(df['data'], desc="特徵轉換進度"):
    dr = ast.literal_eval(i)
    npa = np.asarray(dr, dtype=np.float32)
    dataList.append(npa)

print('\n📥 開始解析勝負標籤...')
for i in tqdm(df['res'], desc="標籤轉換進度"):
    if int(i) == -1:
        resultList.append(0)
    elif int(i) == 1:
        resultList.append(2)
    elif int(i) == 0:
        resultList.append(1)

# ==========================================
# 2. 轉換神經網路輸入格式
# ==========================================
print('\n🔄 正在將資料重塑為神經網路格式...')
NPData = np.array(dataList)
NPResult = np.array(resultList)

NPData = NPData.reshape(-1, 42).astype("float32") 
NPData = NPData.reshape(-1, 6, 7)
NPData = tf.expand_dims(NPData, axis=-1)

# ==========================================
# 3. 建立神經網路架構
# ==========================================
# 這裡的 layers 現在是純粹的 tf_keras.layers，可以完美相容！
print('\n🧠 開始建立並訓練 ResNet 神經網路...')

# 1. 定義輸入層
inputs = keras.Input(shape=(6, 7, 1))
x = data_augmentation_layer(inputs)

# 2. 初始卷積層 (將通道數放大到 64，並提取初步特徵)
# 注意：padding='same' 非常重要，它能保持 6x7 的尺寸不縮水，後續才能順利相加
x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)

# 3. 定義一個「殘差區塊 (Residual Block)」的函數
def residual_block(x_in, filters):
    shortcut = x_in # 備份原始輸入 (也就是殘差架構的精髓)
    
    # 兩層卷積
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(x_in)
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x) # 這裡先不經過 relu
    
    # 💥 將卷積後的結果，加上剛剛備份的原始輸入 (跳層相加)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x) # 相加後再做激活
    return x

# 4. 堆疊殘差區塊 (就像疊積木一樣，你可以自由決定要幾層)
# 這裡先疊加 3 個區塊，對於四子棋來說已經非常強大了
for _ in range(3):
    x = residual_block(x, 64)

# 5. 壓平並輸出
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(3, activation="softmax")(x)

# 6. 把輸入跟輸出綁定成一個模型
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

# ==========================================
# 4. 開始訓練與儲存
# ==========================================
history = model.fit(NPData, NPResult, batch_size=4000, epochs=15, validation_split=0.1)

model.save("monted")
print('\n✅ 訓練完成！全新 AI 大腦已儲存至 monted 資料夾。')