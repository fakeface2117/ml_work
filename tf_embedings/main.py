import numpy as np
import pandas as pd
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split

"""Load 20 newsgroups and save to csv"""
# newsgroups_train = fetch_20newsgroups(subset='train')
# newsgroups_test = fetch_20newsgroups(subset='test')
#
# data = pd.Series(newsgroups_train.data + newsgroups_test.data)
# data = pd.DataFrame(data)
# data.columns = ['text'] + data.columns.tolist()[1:]
# data['target'] = pd.Series(np.concatenate((newsgroups_train.target, newsgroups_test.target)))
#
# data.to_csv('20_news_data.csv', sep=';', index=False)

data = pd.read_csv('20_news_data.csv', sep=';')

df_train, df_test = train_test_split(test_size=0.1)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train["text"])
vocab_size = len(tokenizer.word_index) + 1

x_train = tokenizer.texts_to_sequences(df_train["text"])
x_test = tokenizer.texts_to_sequences(df_test["text"])

# x_train = pad_sequences(x_train, maxlen=max_len)
# x_test = pad_sequences(x_test, maxlen=max_len)


