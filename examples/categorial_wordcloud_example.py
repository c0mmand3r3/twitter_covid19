"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Tuesday, July 13, 2021
"""
import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

if __name__ == '__main__':
    read_path = os.path.join('data', 'original', 'covid19_tweeter_final_dataset.csv')

    data = pd.read_csv(read_path)

    positive_label_data = data.query('Label == 1')
    negative_label_data = data.query('Label == -1')
    neutral_label_data = data.query('Label == 0')

    text = ' '.join(list(positive_label_data['Tweet']))

    wordcloud = WordCloud(width=800, height=800,
                          font_path=os.path.join('data', 'a.otf'),
                          background_color='white',
                          min_font_size=10).generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()