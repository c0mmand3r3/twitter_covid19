"""
-- author: Ashish Mainali
-- email: mainaliashish@outlook.com
-- date: 2021-07-14
"""

import pandas as pd
from collections import Counter

df = pd.read_csv('testdata.csv')


def convert_sentences_to_list(string):
    li = list(string.split(","))
    return li


mergedlist = []

for line in df['Tokanize_tweet']:
    tokens = convert_sentences_to_list(line)
    for token in tokens:
        mergedlist.append(token)

res = Counter(mergedlist).most_common()

common_words_count = pd.DataFrame(res, columns=['words', 'count'])

common_words_count.to_csv('words_frequency.csv')

print('Common words count successfully saved to csv file..')
