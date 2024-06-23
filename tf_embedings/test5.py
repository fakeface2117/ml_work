from keras import datasets, models, layers, utils, backend
import matplotlib.pyplot as plt

backend.clear_session()

"""рекурентные нейронные сети GRU"""

max_words = 10000
max_len = 200

# x = np.ndarray -> [list([1, 2, 3]), list([4, 5, 6]) ...]

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_words)
"""  # padding='post' - означает что нули добавляются в конце.
Но так как это рекурентные нейронки, надо ставить нули в начало"""
x_train = utils.pad_sequences(x_train, maxlen=max_len)
x_test = utils.pad_sequences(x_test, maxlen=max_len)

model = models.Sequential()
model.add(layers.Embedding(max_words, 8, input_length=max_len))
model.add(layers.GRU(32))
model.add(layers.Dense(1, activation='sigmoid'))  # бинарная классификация

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15, batch_size=128, validation_split=0.1)

# посмотерть переобучение
plt.plot(history.history['accuracy'], label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
