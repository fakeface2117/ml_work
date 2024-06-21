import re
import string

import keras
import pandas as pd
import tensorflow as tf
from keras import layers, backend
from sklearn.model_selection import train_test_split
backend.clear_session()
tf.compat.v1.reset_default_graph()
vocab_size = 20000  # максимальная длина словаря
batch_size = 32  # размер батча
epochs = 5  # количество эпох
max_len = 1000  # максимальное количество слов в документе
num_classes = 20

data = pd.read_csv('20_news_data.csv', sep=';')
data['text_len'] = data['text'].apply(lambda text: len(text))
df_train_test, df_val = train_test_split(data, test_size=0.1, random_state=123)
df_train, df_test = train_test_split(df_train_test, test_size=0.1, random_state=132)

training_dataset = tf.data.Dataset.from_tensor_slices(
    (df_train['text'].values, df_train['target'].values)).batch(batch_size=32)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (df_val['text'].values, df_val['target'].values)).batch(batch_size=32)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (df_test['text'].values, df_test['target'].values)).batch(batch_size=32)


def normlize(text):
    remove_regex = f'[{re.escape(string.punctuation)}]'
    space_regex = '...'
    result = tf.strings.lower(text)
    result = tf.strings.regex_replace(result, remove_regex, '')
    result = tf.strings.regex_replace(result, space_regex, ' ')
    return result


tfidf_vectorizer = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='tf-idf',
    ngrams=(2, 3)
)

tfidf_vectorizer.adapt(df_train['text'].values)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return tfidf_vectorizer(text), label


train_ds = training_dataset.map(vectorize_text)
val_ds = validation_dataset.map(vectorize_text)
test_ds = test_dataset.map(vectorize_text)

# model = Sequential()
# model.add(keras.Input(shape=(1,), dtype=tf.string))
# model.add(tfidf_vectorizer)
# model.add(layers.Embedding(vocab_size, 128))
# model.add(layers.Dropout(0.5))
# model.add(layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3))
# model.add(layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(128, activation="relu"))
# model.add(layers.Dropout(0.5))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
inputs = keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(vocab_size, 128)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(train_ds, validation_data=val_ds, epochs=epochs)
q = model.evaluate(test_ds)
print(q)

# history = model.fit(
#     training_dataset,
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=1,
#     validation_data=validation_dataset,
#     validation_split=0.1
# )

# def vectorize_text(text, label):
#     text = tensorflow.expand_dims(text, -1)
#     return vectorize_layer(text), label

# inputs = keras.Input(shape=(None,), dtype="int64")
#
# text_input = keras.Input(shape=(1,), dtype=tensorflow.string, name='text')
# x = vectorize_layer(text_input)
# x = layers.Embedding(max_words, 128)(x)
# x = layers.Dropout(0.5)(x)
# x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
# x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
# x = layers.GlobalMaxPooling1D()(x)
# x = layers.Dense(128, activation="relu")(x)
# x = layers.Dropout(0.5)(x)
#
# # We project onto a single unit output layer, and squash it with a sigmoid:
# predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
#
# model = keras.Model(inputs, predictions)
#
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
#
# history = model.fit(
#     x_train, y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=1,
#     validation_split=0.1
# )
# score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
# print('\n')
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
# results = model.predict(x_test, batch_size=batch_size, verbose=1)
