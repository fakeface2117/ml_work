import numpy as np
import pandas as pd

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


