"""
-- author: Ashish Mainali
-- email: ashishmainalee@gmail.com
-- date: 2021 January 14
"""

import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    read_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_Stemmer_tokenize.csv')
    write_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_analysis.csv')

    df = pd.read_csv(read_path)

    words_list = ' '.join(df['Stemmer_tokens']).replace(";", ",").replace(" ", ",").split(',')
    common_words = Counter(words_list).most_common(None)

    common_words_count = pd.DataFrame(common_words, columns=['words', 'count'])
    common_words_count.to_csv(write_path)
    print(f'Common words successfully extracted and saved to {write_path}')

    ten_common_words = Counter(words_list).most_common(10)
    df_common_words = pd.DataFrame(ten_common_words, columns=['words', 'count'])

    plt.rc('font', family="Lohit Devanagari, Arial")
    fig, ax = plt.subplots(figsize=(9, 9))
    # Plot horizontal bar graph
    df_common_words.sort_values(by='count').plot.barh(x='words',
                                                           y='count',
                                                           ax=ax,
                                                           color="blue")
    ax.set_title("Most Frequent Tokens Found in Tweets")

    plt.show()
