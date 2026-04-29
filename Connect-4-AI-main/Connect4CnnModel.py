from statistics import mode
from turtle import shape
import pandas as pd
import ast
import numpy as np
import mnist
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers,models, regularizers
import dask.dataframe as dd
from tensorflow.keras.models import Sequential

# ###################### 


df = dd.read_csv('final.csv')
df = df.dropna()

dataList = []
resultList = []

for i in df['data']:
    dr = ast.literal_eval(i)
    npa = np.asarray(dr, dtype=np.float32)
    dataList.append(npa)

print('first Data in')

for i in df['res']:
    if int(i) == -1:
        resultList.append(0)
    if int(i) == 1:
        resultList.append(2)
    if int(i) == 0:
        resultList.append(1)

print('second Data in')

# Yellow win is 1, Red win is -1

NPData = np.array(dataList)
NPResult = np.array(resultList)

NPData = NPData.reshape(4000000,42).astype("float32") 

print('making text files')

np.savetxt('NPData.txt', NPData, fmt='%d')
np.savetxt('NPResult.txt', NPResult, fmt='%d')

NPData = np.loadtxt('NPData.txt')
NPResult = np.loadtxt('NPResult.txt')

NPData = NPData.reshape(4000000,6,7)
# Basically making images of 6,7 with 1 channel.
NPData = tf.expand_dims(NPData, axis=-1)

data_augmentation_layer = tf.keras.Sequential([                                    
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')
], name='data_augmentation')

model = keras.Sequential(
    [
        keras.Input(shape=(6,7,1)),
        data_augmentation_layer,
        layers.Conv2D(64, kernel_size=(4, 4), activation="relu", padding="valid"),
        # layers.MaxPooling2D((4, 4),padding='same'),
        layers.Conv2D(128, kernel_size=(2, 2), activation="relu",padding='valid'),
        # layers.MaxPooling2D((2, 2),padding='same'),
        # layers.Conv2D(64, kernel_size=(2, 2), activation="relu",padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation="softmax"),
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history = model.fit(NPData, NPResult, batch_size=4000, epochs=15, validation_split=0.1)

model.save("monted")

# model = keras.models.load_model("monted")


