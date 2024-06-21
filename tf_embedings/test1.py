from keras import datasets, models, layers, utils, preprocessing
import numpy as np
import matplotlib as plt

"""Набор данных уже закодирован в числа"""

max_words = 10000
max_len = 200

# x = np.ndarray -> [list([1, 2, 3]), list([4, 5, 6]) ...]

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_words)
x_train = utils.pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = utils.pad_sequences(x_test, maxlen=max_len, padding='post')

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(max_len,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # 1 потому что бинарная классификация

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.1)
