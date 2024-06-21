from keras import datasets, models, layers, utils, preprocessing, backend
import numpy as np
import matplotlib as plt

backend.clear_session()

"""Использование one hot encoding векторизации"""

max_words = 10000
max_len = 200

# x = np.ndarray -> [list([1, 2, 3]), list([4, 5, 6]) ...]

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_words)


# тексту уже соответствует набор чисел
def vectorize_sequences(sequences: np.ndarray, dimension: int):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# векторизация данных OHE
x_train = vectorize_sequences(x_train, max_words)
x_test = vectorize_sequences(x_test, max_words)

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(max_words,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # 1 потому что бинарная классификация

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
