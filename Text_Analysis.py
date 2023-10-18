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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud

import pandas as pd
from PIL import Image
import numpy as np
import random
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

#%% Improving word cloud

#Define gray_color_func function and mask variable for advanced word cloud
mask = np.array(Image.open("man_in_top_hat.jpeg"))

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

#Create advanced Word Cloud with our text data
wordcloud = WordCloud(background_color="purple", mask=mask, color_func=grey_color_func, max_words=100, stopwords=stopwords_wc, random_state=1).generate(word_cloud_text)
plt.figure(figsize=(12, 9))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#%% Word Frequency Analysis (AKA, something that's not a complete distraction)

#Word Frequency Distribution
fdist = nltk.FreqDist(tokens)
fdist

#%% 50 Most Common words

fdist.most_common(50)

#%% Visualization of top 50 most common words in text

plt.figure(figsize=(12,6))
fdist.plot(50)
plt.show()

#%% Cumulative word count

plt.figure(figsize=(12,6))
fdist.plot(50,cumulative=True)
plt.show()

#%% Vader Sentiment Analysis

#Initialize Vader sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

#Perform Vader Sentiment Analysis
sentences['compound'] = [analyzer.polarity_scores(x)['compound'] for x in sentences['sentence']]
sentences['neg'] = [analyzer.polarity_scores(x)['neg'] for x in sentences['sentence']]
sentences['neu'] = [analyzer.polarity_scores(x)['neu'] for x in sentences['sentence']]
sentences['pos'] = [analyzer.polarity_scores(x)['pos'] for x in sentences['sentence']]
sentences.head(10)

#%% Vader Sentiment Analysis Visualization

#Get number of positive, neutral, and negative sentences
positive_sentence = sentences.loc[sentences['compound'] > 0]
neutral_sentence = sentences.loc[sentences['compound'] == 0]
negative_sentence = sentences.loc[sentences['compound'] < 0]

print(sentences.shape)
print(len(positive_sentence))
print(len(neutral_sentence))
print(len(negative_sentence))

#Plot Histogram
plt.figure(figsize=(14,6))
plt.hist(sentences['compound'], bins=50);

#%% Topic Modelling

#Convert sentence data to list
data = sentences['sentence'].values.tolist()
type(data)

#Text cleaning and tokenization using function
def text_processing(texts):
    # Remove numbers and alphanumerical words we don't need
    texts = [re.sub("[^a-zA-Z]+", " ", str(text)) for text in texts]
    # Tokenize & lowercase each word
    texts = [[word for word in text.lower().split()] for text in texts]
    # Stem each word
    lmtzr = WordNetLemmatizer()
    texts = [[lmtzr.lemmatize(word) for word in text] for text in texts]
    # Remove stopwords
    stoplist = stopwords.words('english')
    texts = [[word for word in text if word not in stoplist] for text in texts]
    # Remove short words less than 3 letters in length
    texts = [[word for word in tokens if len(word) >= 3] for tokens in texts]
    return texts

# Apply function to process data and convert to dictionary
data = text_processing(data)
dictionary = Dictionary(data)
len(dictionary)

#Create corpus for LDA analysis
corpus = [dictionary.doc2bow(text) for text in data]
len(corpus)

#%% Latent Dirichlet Allocation (long runtime)

#Find optimal k value for the number of topics for our LDA analysis
np.random.seed(1)
k_range = range(6,20,2)
scores = []

for k in k_range:
    LdaModel = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=20)
    cn = CoherenceModel(model = LdaModel, corpus=corpus, dictionary=dictionary,coherence="u_mass")
    print(cn.get_coherence())
    scores.append(cn.get_coherence())

plt.figure()
plt.plot(k_range,scores)

# "Optimal" is verbage chosen by course. Coherence seems to be monotonically
# decreasing, leaving choosing k to be a judgement call.

#%% Build LDA topic model

model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=4,passes=20)

model.show_topics()

# Topics are like correlated word clusters. Not super clear how to leverage
# this to make an analytical judgement without already knowing the text, tbh















