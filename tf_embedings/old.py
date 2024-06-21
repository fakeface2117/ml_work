import pandas as pd
import re
from string import punctuation
import nltk
from pymorphy3 import MorphAnalyzer
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout

russian_stopwords = nltk.corpus.stopwords.words("russian")
russian_stopwords.extend(['это', 'нею', 'ый', 'т', 'х'])

morph = MorphAnalyzer()

regular_str = '[^А-Яа-я0-9:,.\-\n ]'
spec_chars = punctuation + '\n\xa0«»\t—…'


def remove_chars_from_text(text: str, chars: str) -> str:
    return "".join([ch for ch in text if ch not in chars])


def remove_sw(text: list) -> list:
    return [i for i in text if i not in russian_stopwords]


def lematization(text: list) -> str:
    out_text = ''
    for i in text:
        normal_word = morph.normal_forms(i.strip())[0]
        if len(normal_word) > 3:
            out_text += (normal_word + ' ')
    return out_text.strip()


def clear_data(df: pd.DataFrame):
    df2: pd.DataFrame = df.loc[:, ['text', 'category']]
    df2['text'] = df2['text'].apply(lambda x: x.lower())
    digits_seq = '\d'
    df2['text'] = df2['text'].apply(lambda x: re.sub(digits_seq, '', x))
    df2['text'] = df2['text'].apply(lambda x: re.sub(regular_str, '', x))
    df2['text'] = df2['text'].apply(lambda x: remove_chars_from_text(x, spec_chars))
    df2['text'] = df2['text'].apply(lambda x: nltk.word_tokenize(x))
    df2['text'] = df2['text'].apply(lambda x: remove_sw(x))
    df2['text'] = df2['text'].apply(lambda x: lematization(x))

    return df2


"""dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_dataset = dataset['train']
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE)  # .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

z = 1
for example, label in train_dataset:
    print()
    print("z: ", z)
    print('texts: ', example.numpy())
    print('labels: ', label.numpy())
    z += 1

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
# .adapt метод устанавливает словарный запас слоя. Вот первые 20 жетонов. После заполнения и неизвестных токенов они сортируются по частоте:
encoder.adapt(train_dataset.map(lambda text, label: text))
vocab = np.array(encoder.get_vocabulary())

for example, label in train_dataset:
    encoded_example = encoder(example).numpy()
    print(encoded_example)"""

data = pd.read_csv('c:\\work\\train_clear.csv', sep=';')

categories = {}
for key, value in enumerate(data['category'].unique()):
    categories[value] = key
data['label'] = data['category'].map(categories)

data = data.sample(frac=1).reset_index(drop=True)

max_words = 0
for desc in data['text']:
    words = len(desc.split())
    if words > max_words:
        max_words = words

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'].tolist())

text_sequences = tokenizer.texts_to_sequences(data['text'].tolist())

x_train, x_test, y_train, y_test = train_test_split(
    text_sequences,
    data['label'],
    test_size=0.2
)

total_words = len(tokenizer.word_index)

num_words = 5000

print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=num_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = to_categorical(y_train, 13)
y_test = to_categorical(y_test, 13)

# количество итераций
epochs = 10

print(u'Собираем модель...')
model = Sequential()
model.add(Dense(512, input_shape=(num_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(13))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

fited_model = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=epochs,
                        verbose=1)

score = model.evaluate(x_test, y_test,
                       batch_size=32, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))
