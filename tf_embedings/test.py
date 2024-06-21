import pandas as pd
import tensorflow as tf
from keras.src.legacy.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras import Sequential, preprocessing
from keras import layers, backend
backend.clear_session()
tf.compat.v1.reset_default_graph()

df_init = pd.read_csv('train_clear.csv', sep=';', encoding='utf8')
# df_source = df_init[['text', 'category']]

texts = df_init['text'].values
categories = {}
for key, value in enumerate(df_init['category'].unique()):
    categories[value] = key
df_init['label'] = df_init['category'].map(categories)
labels = df_init['label'].values


texts_train, texts_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=1000
)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_train)

x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=5,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    batch_size=10)

loss_train, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss_test, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
