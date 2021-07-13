**Scrapping Date**
<br/>
<br/>
From : 2020-2-11 To : 2021-1-10
<br/>
File path : - data/twitter_dataset/raw-covid-tweets.csv
<br/>
Python Run File Path : - examples/twitter_scrapping_example.py
<br/>


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
<h6>CSV file Information </h6>

`First File Path: - data/preprocessing/cleaning_data/covid_tweets_clean.csv`
<br/>
`Second File Path: - data/preprocessing/cleaning_data/covid_tweets_tokenize.csv`

<h6>Python file Information </h6>
`First Python File: - examples/raw_tweets_filter_example.py`
<br/>
`Second Python File: - examples/remove_duplicate_tweets_example.py`
<br/>
`Third Python File: - examples/remove_stopping_words_example.py`



<h1>Step 2</h1>
<h3>Stemmer data</h3>

The Stemmer is written in Java. Thus, csv file is used as a data source for stemmer.
<br/>
<br/>
`Data source: - data/preprocessing/cleaning_data/covid_tweets_tokenize.csv`
<br/>
`Generate Data Path: - data/preprocessing/stemmer_data/covid_tweets_Stemmer_tokenize.csv`
<br/>


<h1>Step 3</h1>
<h3>Data Preparation Annotation Manually</h3>

The annotation is prepred by four different author.
<hr/>
Prepared By: <hr/>
1-18000 : Ashish Mainali
18000-26000 : Anish Basnet
26000-end : Tej Bahadur Shahi
Review : Chiranjibi Sitaula

<h1>Step 4</h1>
<h3> Covid Dataset collection</h3>

First file to run : 
 - data_collection_example.py

Second file  to run:
 - covid19_query_dataset_example.py

<hr/>
First file helps to collect the data from two different csv file.

<hr/>
Second file help to filtered out unnecessary labeling error.
<hr/>
Total Tweets : 33,473 <hr/>
Labeling Error Tweets : 38 <hr/>
Final Total Tweets : 33,435 <hr/>
Final Dataset Name : covid19_tweeter_final_dataset.csv <hr/>
