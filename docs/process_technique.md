**Scrapping technique**

Scrap from Twitter <br/>
Time limit of the scrapping link scroll down should be given.
<br/>
`for example, sub_scrap.get_page(pause_time=10, activate_scroll_down=True)`
<br/>
Pause time can be changed with respect to seconds.


**Scrapping Date**
<br/>
<br/>
From : 2020-2-11 To : 2021-1-10
<br/>
File path : - data/twitter_dataset/raw-covid-tweets.csv

<h1>Step 1</h1>
<h3>Cleaning of data</h3>

**steps to follow**
<br/>
1 - Remove duplicate tweets and replace with single tweets.
<br/>
2 - Duplicate tweets datetime are multiple. Thus, it needs to choose the earliest datetime as possible. 
<br/>
3 - Tokenize the tweets at word level.
<br/>
4 - Remove the stop words if there exist. (stop words list is attached below) 
<br/>
5 - Data should be written in csv format.
<br/>
The format of tweets should be- 
<br/>
Datetime 
<br/>
Tweets (full content)
<br/>
Tokenize tweets ( seperated by comma) 
<br/>
Eg. <br/>
0,2021-01-10 22:06:41+00:00,अमेरिकामा कोभिड बाट एकै दिन चार हजारभन्दा बढीको मृत्यु, ' अमेरिकामा,कोभिड,बाट,एकै,दिन,चार,हजारभन्दा,बढीको,मृत्यु'

`First File Path: - data/preprocessing/cleaning_data/covid_tweets_clean.csv`
<br/>
`Second File Path: - data/preprocessing/cleaning_data/covid_tweets_tokenize.csv`




<h1>Step 2</h1>
<h3>Stemmer data</h3>