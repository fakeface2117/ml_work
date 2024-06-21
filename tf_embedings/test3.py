from keras import datasets, models, layers, utils, backend
import matplotlib.pyplot as plt
backend.clear_session()

"""Использование embedings векторизации"""

max_words = 10000
max_len = 200

# x = np.ndarray -> [list([1, 2, 3]), list([4, 5, 6]) ...]

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_words)
x_train = utils.pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = utils.pad_sequences(x_test, maxlen=max_len, padding='post')

model = models.Sequential()
model.add(layers.Embedding(max_words, 2, input_length=max_len))
model.add(layers.Dropout(0.25))  # снижает вероятность переобучения
model.add(layers.Flatten())  # делает плоский вектор
model.add(layers.Dense(1, activation='sigmoid'))  # бинарная классификация

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15, batch_size=128, validation_split=0.1)

scores = model.evaluate(x_test, y_test, verbose=1)

# получить слой с вложениями  (веса embedings)
embedings_matrix = model.layers[0].get_weights()[0]

word_index_org: dict = datasets.imdb.get_word_index()

word_index = dict()
for word, number in word_index_org.items():
    word_index[word] = number + 3
word_index['<Заполнитель>'] = 0
word_index['<Начало последовательности>'] = 1
word_index['<Неизвестное слово>'] = 2
word_index['<Не используется>'] = 3

# отрисовать embedings
something_words = ['brilliant', 'fantastic', 'amazing', 'good', 'bad', 'awful', 'trash', 'ugly']
int_words = []
for word in something_words:
    int_words.append(word_index[word])

review_vectors = embedings_matrix[int_words]

plt.scatter(review_vectors[:, 0], review_vectors[:, 1])
for i, txt in enumerate(int_words):
    plt.annotate(txt, (review_vectors[i, 0], review_vectors[i, 1]))

# сохранить вектора в файл
reverse_word_index = dict()
for k, v in word_index.items():
    reverse_word_index[v] = k

with open('saved_embedings.csv', 'w') as f:
    for word_num in range(max_words):
        word = reverse_word_index[word_num]
        vec = embedings_matrix[word_num]
        f.write(word + ',')
        f.write(','.join([str(x) for x in vec]) + '\n')
