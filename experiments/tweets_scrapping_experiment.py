"""
- author: Ashish Mainali
- email: ashishmainalee@gmail.com
- date: 2021-01-11
"""

# Imports
import snscrape.modules.twitter as sntwitter
import pandas as pd

if __name__ == '__main__':

    # Creating list to append tweet data to
    tweets_list = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('कोभिड-१९ since:2019-11-28 until:2021-01-11').get_items()):
        tweets_list.append([tweet.date, tweet.content])

    # Creating a dataframe from the tweets list above
    tweets_df2 = pd.DataFrame(tweets_list, columns=['Datetime', 'Text'])

    # Display first 5 entries from dataframe
    tweets_df2.head()

    # Export dataframe into a CSV
    tweets_df2.to_csv('covid-tweets.csv', sep=',', index=False)