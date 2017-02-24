import sys
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import requests
import prettytable
import nltk
from SentimentIngress import *

predictSentiment("review_polarity/txt_sentoken/", "")