#NYTSentiment
#Sentiment analysis for the NYT
#By Dieter Brehm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
import requests
import nltk
import pandas as pd
import numpy as np
import re
import json

def makeJsonRequest(queryText):
    search = queryText
    TimesKey = ""
    url="http://api.nytimes.com/svc/search/v2/articlesearch.json"
    query_params={'q' : search,
                  'api-key': TimesKey,
         }
    request = requests.get(url, params=query_params)
    data = request.json()
    return data

def AnalyzeSentiment(testDataFile, ):
    #should be a csv file
    #pandas really likes csv files for use in dataframes
    testDataPath = testDataFile

    #load in a dataset into both testing and training data frames
    test_data_frame = pd.read_csv(testDataPath, header=None, delimiter="\t", quoting=3 )
    test_data_frame.columns = ["Text"]
    train_data_frame = pd.read_csv('original_train_data.csv', header=None, delimiter="\t", quoting=3 )
    train_data_frame.columns = ["Sentiment", "Text"]

    #strip all the useless punctuation and other unnecessary information
    stemmer = PorterStemmer()
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    #make the words seperate and further remove useless items
    def tokenize(text):
        text = re.sub("[^a-zA-Z]", " ", text)
        #tokens are basically seperate instances of words
        #with them we can do things like count instances of specific words
        #or graph average number of words per row(if there are multiple rows of text :P
        tokens = nltk.word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems

    #take our tokenize function and use it on our training data set
    vectorizer = CountVectorizer(
        analyzer= 'word',
        tokenizer= tokenize,
        lowercase= True,
        stop_words= 'english',
        max_features= 85
    )

    #combine all the data we need to look at, training and to be predicted, into one collection
    fit_data = vectorizer.fit_transform(train_data_frame.Text.tolist() + test_data_frame.Text.tolist())
    fit_data = fit_data.toarray()

    #predict the sentiment of unlabeled data
    model = LogisticRegression()
    model = model.fit(X=fit_data[0:len(train_data_frame)], y=train_data_frame.Sentiment)
    pred = model.predict(fit_data[len(train_data_frame):])

    #read each row of the text column along with the predicted sentiment
    #1 is positive
    #0 is negative
    for text, sentiment in zip(test_data_frame.Text, pred):
        print(sentiment, text)

