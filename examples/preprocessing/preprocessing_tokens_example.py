"""
- author: Ashish Mainali
- email: ashishmainalee@gmail.com
- date: 2021-01-13
"""

import os
import pandas as pd
from config import Config
from tweeter_covid19.utils import read_file, process_rasuwa_dirga, preprocess_nepali_documents

if __name__ == '__main__':
    stopping_words_path = Config.get_instance()['stop_word_path']
    read_path = os.path.join('data', 'original', 'covid19_tweeter_final_dataset.csv')
    write_path = os.path.join('data', 'original', 'covid19_tweeter_final_dataset_clean_data.csv')

    raswa_dirga_file_path = Config.get_instance()['raswa_dirga_pair_path']

    stop_words = read_file(stopping_words_path)
    pairs_words = read_file(raswa_dirga_file_path)
    tweets = pd.read_csv(read_path)

    data_df = dict({
        'Label': [],
        'Datetime': [],
        'Tweet': [],
        'Tokanize_tweet': []
    })

    for index, tweet in enumerate(tweets['Tweet']):
        tokens = preprocess_nepali_documents([tweet], stop_words)
        tokens = process_rasuwa_dirga(tokens=tokens, pairs=pairs_words, verbose=True)
        clean_tweets = []
        for _index, token in enumerate(tokens):
            if not token[0] in stop_words:
                clean_tweets.append(token[0])
        data_df['Label'].append(tweets['Label'][index])
        data_df['Datetime'].append(tweets['Datetime'][index])
        data_df['Tweet'].append(' '.join(clean_tweets))
        data_df['Tokanize_tweet'].append(','.join(clean_tweets))
    my_df = pd.DataFrame(data_df)
    my_df.to_csv(write_path)

    print("completed")
