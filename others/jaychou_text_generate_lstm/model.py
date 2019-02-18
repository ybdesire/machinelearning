#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding
from keras import optimizers
from keras.callbacks import ModelCheckpoint


with open('jaychou_lyrics.txt' ,'r', encoding='utf-8') as fr:
    corpus_chars = fr.read()

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]#只取前10000个词


idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)# 有多少个独立的字符

seq_length = 100  # 句子长度, 根据100个词，预测第101个词
n_words = len(corpus_chars)
x_data = []
y_data = []
for i in range(0, n_words - seq_length, 1):
    seq_in = corpus_chars[i:i + seq_length]
    seq_out = corpus_chars[i + seq_length]
    x_data.append([char_to_idx[char] for char in seq_in])
    y_data.append(char_to_idx[seq_out])
    
x_data = np.array(x_data)    
y_data = np.array(y_data)
print(x_data.shape, y_data.shape)
y_data = np_utils.to_categorical(y_data)
print(x_data.shape, y_data.shape)


# model
model = Sequential()
model.add(Embedding(vocab_size, 512, input_length=seq_length))
model.add(LSTM(512, input_shape=(seq_length, 512), return_sequences=True))
model.add(LSTM(1024))
model.add(Dense(vocab_size, activation='softmax'))

# train
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam)

# 存储每一次迭代的网络权重
filepath = "weights-improvement={epoch:02d}-{loss:4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x_data, y_data, epochs=50, batch_size=100, callbacks=callbacks_list, verbose=1)


