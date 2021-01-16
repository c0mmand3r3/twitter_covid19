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
    write_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_lda.csv')

    df = pd.read_csv(read_path)

    words_list = ' '.join(df['Stemmer_tokens']).replace(";", ",").replace(" ", ",").split(',')
    # print([words_list])

    # Create Dictionary from the articles
    dictionary = corpora.Dictionary([words_list])

    # print(dictionary)
    # exit(0)
    # Create document term matrix
    doc_term_matrix = [dictionary.doc2bow(word) for word in [words_list]]
    # print(doc_term_matrix[0])

    # Instantiate and Fit data into LDA model
    lda_model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=10, id2word=dictionary)
    topics_lda = lda_model.print_topics(num_topics=10)
    for item in topics_lda:
        print(item)
