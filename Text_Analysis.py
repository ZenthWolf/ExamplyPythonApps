#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:42:29 2023

@author: zenth
"""

#%% Imports

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.stem import WordNetLemmatizer

#from gensim.corpora import Dictionary
#from gensim.models import ldamodel
#from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud

import pandas as pd
#from PIL import Image
#import numpy as np
#import random
import re

import matplotlib.pyplot as plt
#External Window
#%matplotlib qt
#Inline
#%matplotlib inline

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%% NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

#%% Read in book text
textfile = open('great_expectations.txt', 'r', encoding='utf-8')
great_expect = textfile.read()
#print(great_expect)

#%% Cleaning Text Data

#Lowercase words for word cloud
word_cloud_text = great_expect.lower()
#Remove numbers and alphanumeric words we don't need for word cloud
word_cloud_text = re.sub("[^a-zA-Z0-9]", " ", word_cloud_text)


#Tokenize the data to split it into words
tokens = word_tokenize(word_cloud_text)
#Remove stopwords
tokens = (word for word in tokens if word not in stopwords.words("english"))
#Remove short words less than 3 letters in length
tokens = (word for word in tokens if len(word) >=3)

#Data cleaning to split data into sentences
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

text = " " + great_expect + "  "
text = text.replace("\n"," ")
text = re.sub(prefixes,"\\1<prd>",text)
text = re.sub(websites,"<prd>\\1",text)
text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
if "..." in text: text = text.replace("...","<prd><prd><prd>")
if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
if "”" in text: text = text.replace(".”","”.")
if "\"" in text: text = text.replace(".\"","\".")
if "!" in text: text = text.replace("!\"","\"!")
if "?" in text: text = text.replace("?\"","\"?")
text = text.replace(".",".<stop>")
text = text.replace("?","?<stop>")
text = text.replace("!","!<stop>")
text = text.replace("<prd>",".")
sentences = text.split("<stop>")
sentences = [s.strip() for s in sentences]
sentences = pd.DataFrame(sentences)
sentences.columns = ['sentence']

#Print out sentences variable
print(len(sentences))
#sentences.head(10)

#Remove the first few rows of text that are irrelevant for analysis
sentences.drop(sentences.index[:59], inplace=True)
sentences = sentences.reset_index(drop=True)
sentences.head(10)

#%% Create word cloud with our text data

stopwords_wc = set(stopwords.words("english"))

wordcloud = WordCloud(max_words=100, stopwords=stopwords_wc, random_state=1).generate(word_cloud_text)
plt.figure(figsize=(12,16))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
