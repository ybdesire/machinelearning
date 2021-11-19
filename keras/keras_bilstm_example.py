from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import numpy as np
from keras.utils import to_categorical

# example data
X_train = np.zeros((1000,250))
Y_train = np.zeros(1000)
Y_train[:100]=1
print(X_train.shape, Y_train.shape)
Y_train = to_categorical(Y_train)
print(X_train.shape, Y_train.shape)

X_test = np.zeros((300,250))
Y_test = np.zeros(300)
Y_test[:100]=1
print(X_test.shape, Y_test.shape)
Y_test = to_categorical(Y_test)
print(X_test.shape, Y_test.shape)
'''
(1000, 250) (1000,)
(1000, 250) (1000, 2)
(300, 250) (300,)
(300, 250) (300, 2)
'''

# para
MAX_NB_WORDS = 500
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 120

# model 
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 3
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))


'''
Train on 1000 samples, validate on 300 samples
Epoch 1/3
1000/1000 [==============================] - 6s 6ms/step - loss: 0.4372 - accuracy: 0.8990 - val_loss: 0.7150 - val_accuracy: 0.6667
Epoch 2/3
1000/1000 [==============================] - 5s 5ms/step - loss: 0.3360 - accuracy: 0.9000 - val_loss: 0.8327 - val_accuracy: 0.6667
Epoch 3/3
1000/1000 [==============================] - 5s 5ms/step - loss: 0.3248 - accuracy: 0.9000 - val_loss: 0.8641 - val_accuracy: 0.6667
'''

