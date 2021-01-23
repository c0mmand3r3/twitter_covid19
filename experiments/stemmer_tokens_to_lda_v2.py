"""
-- Author: Ashish Mainali
-- E-mail: ashishmainalee@gmail.com
-- Date: 2021 January 22
"""

import os

import gensim
import pandas as pd
from gensim import corpora, models

if __name__ == '__main__':
    read_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_Stemmer_tokenize.csv')

    # Read data into dataframe
    df = pd.read_csv(read_path)

    # Preparing Tokens for LDA analysis
    all_tokens = []
    for index, tokens in enumerate(df['Stemmer_tokens']):
        tokens = tokens.replace(";", ",")
        tokens = tokens.split(",")
        all_tokens.append(tokens)

    # Create Dictionary from the articles
    dictionary = corpora.Dictionary(all_tokens)

    # print(dictionary)
    # exit(0)

    # Create document term matrix
    doc_term_matrix = [dictionary.doc2bow(word) for word in all_tokens]
    # print(doc_term_matrix)
    # exit(0)

    # Fed Tokens into LDA Model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix,
                                                id2word=dictionary,
                                                num_topics=10,
                                                random_state=100,
                                                update_every=1,
                                                passes=60,
                                                alpha='auto',
                                                per_word_topics=True)

    topics_lda = lda_model.print_topics(num_topics=10)

    # Print Top 10 topics with each topic having 10 topics.
    for index, item in topics_lda:
        # print('Slice : {} - \nTop topic: {}.\n'.format(index + 1, item))
        print(f'Slice - {index+1} \n Top topic - {item}')

    print('Top 10 topics generated Successfully..')
    exit(0)
