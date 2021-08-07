"""
-- author: Ashish Mainali
-- email: ashishmainalee@gmail.com
-- date: 2021 January 16
"""

import os

import gensim
import pandas as pd
from gensim import corpora, models

if __name__ == '__main__':
    read_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_Stemmer_tokenize.csv')
    # write_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_lda.csv')

    df = pd.read_csv(read_path)

    words_list = ' '.join(df['Stemmer_tokens']).replace(";", ",").replace(" ", ",").split(',')
    # print(len(words_list))

    topic_tokens_len = int(len(words_list) / 10)
    # print(topic_tokens_len)
    # exit(0)
    initial_token = 0
    final_topic_token = topic_tokens_len
    lda_results = []
    # new_word_list = []



    for i in range(1, 10, 1):
        new_word_list = words_list[initial_token:final_topic_token]
        # print(type(new_word_list))
        # exit(0)
        # print(len(new_word_list))
        # exit(0)
        # print(new_word_list)
        dictionary = corpora.Dictionary([new_word_list])
        # print(dictionary)
        # exit(0)
        doc_term_matrix = [dictionary.doc2bow(word) for word in [new_word_list]]
        # print(doc_term_matrix)
        # exit(0)
        lda_model = gensim.models.ldamodel.LdaModel(doc_term_matrix,
                                                    id2word=dictionary,
                                                    num_topics=10,
                                                    update_every=1,
                                                    random_state=0,
                                                    chunksize=100,
                                                    passes=50,
                                                    alpha='auto',
                                                    per_word_topics=True)
        lda_results.append(lda_model.print_topics(num_topics=1))
        # new_word_list = []
        # print(lda_results)
        # exit(0)
        initial_token += topic_tokens_len
        # print(topic_tokens_len)
        # print(len(words_list))

        if topic_tokens_len < len(words_list):
            final_topic_token += topic_tokens_len

    for index, item in enumerate(lda_results):
        print('Slice : {} - \nTop topic: {}.\n'.format(index + 1, item))
    print('Completed..')
    exit(0)
