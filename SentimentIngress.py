#NYTSentiment
#Sentiment analysis for the NYT
#By Dieter Brehm
#Training code adapted from github user bonzanini


import sys
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import requests
import prettytable
import nltk

def predictSentiment(root_directory, data_input):
    # import the data from a specified directory
    directory = root_directory

    # for eventual feature analysis
    classes = ['pos', 'neg']
    # for the learning
    train_data = []
    train_labels = []

    # for measuring some accuracy
    #needs to be a list
    test_data = []
    #test_data.append(data_input)
    #test_labels = []
    #put our dataset into usable classifications
    #dataset has the following structure
    #root folder
    #->negative folder
    #->->negative text files
    #->positive folder
    #->-positive text files
    for current in classes:
        current_dir = os.path.join(directory, current)
        #movie reviews are generally either negative OR positive
        #listdir gives a list of the reviews in a directory
        for review in os.listdir(current_dir):
            #with is essentially a try statement, but simpler to use. Tries to read
            #text files
            with open(os.path.join(current_dir, review), 'r') as file:
                content = file.read()
                #most of the set is for training, but we need something to test against!
                if review.startswith('thisSucks'):
                    test_data.append(content)
                else:
                    train_data.append(content)
                    train_labels.append(current)
    #for our training process, we are considering words to be features
    #This will use a library to automatically weight words based on frequency
    #It will filter based on max and min frequency
    print(test_data)
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df =0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    #let's try to classify some text
    classifier = svm.SVC()
    classifier.fit(train_vectors, train_labels)
    prediction = classifier.predict(test_vectors)

    print(prediction)