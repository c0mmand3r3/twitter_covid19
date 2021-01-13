"""
- author: Ashish Mainali
- email: ashishmainalee@gmail.com
- date: 2021-01-12
"""

import os
import pandas as pd

if __name__ == '__main__':
    read_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_clean.csv')
    write_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_no_duplicate.csv')

    df = pd.read_csv(read_path)

    new_text = dict({'Datetime': [], 'Tweets': [], 'Tokanize_tweets': []})
    df.drop_duplicates(subset='Clean_text', inplace=True, keep='last')
    tweets_df = df.reset_index(drop=True)
    tweets_df.head()

    for index, content in enumerate(tweets_df['Clean_text']):
        new_text['Datetime'].append(tweets_df['Datetime'][index])
        new_text['Tweets'].append(content)
        separated_content = content.replace(' ', ',')
        new_text['Tokanize_tweets'].append(separated_content)
        print(str(index)+' '+separated_content)

    tweets_df = pd.DataFrame(new_text)
    tweets_df.to_csv(write_path)
    print("Tweets Separated and Duplicate tweets removed Successfully...")


    # published_dates = []
    # for index, content in enumerate(df['Clean_text']):
    #     for [index, heading] in enumerate(df['Clean_text']):
    #         if df['Clean_text'][index] == heading:
    #             print(df['Clean_text'][index])
    #             print(heading)
    #             published_dates.append(df['Datetime'][index])
    #             print(published_dates)
    #             index += 1
    #             # print(index)
    #             exit(0)
    #             # print(df['Datetime'][index])
    #             # duplicate_text = published_dates.append((df['Datetime']))
    #             # print(duplicate_text)
    #         else:
    #             exit(0)
    #             # df.to_csv(write_path)

