# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for bitcoin_price_predictor Project
"""

from os.path import split
import pandas as pd
import numpy as np
import datetime
import ast
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

nltk.download('averaged_perceptron_tagger')
#pd.set_option('display.width', 200)


def clean_coins_colmns(data):
    #TO DO: Need to adjust this!
    #top_bitcoin_news["clean_coins"]=list(map(lambda x:ast.literal_eval(x)[0]["name"], top_bitcoin_news["coins"]))

def clean_date_columns(data):
    #TO DO: Need to adjust this!
    # top_bitcoin_news["date"]=list(map(lambda x: pd.to_datetime(x[:10], format='%Y/%m/%d'), top_bitcoin_news["publishedAt"]))

def clean_text(text):
    text = text.lower()
    text = ''.join(word for word in text if not word.isdigit())
    text = ' '.join(token.lower() for token in word_tokenize(text) if token.lower() not in stop_words)
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    lemmatizer = WordNetLemmatizer()
    text = ''.join(lemmatizer.lemmatize(word) for word in text)

    return text


def sentimenter(text):
    blob = TextBlob(text)
    blob.tags
    blob.noun_phrases
    for sentence in blob.sentences:
        return(sentence.sentiment.polarity)



if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    import bitcoin_price_predictor
    folder_source, _ = split(bitcoin_price_predictor.__file__)
    df = pd.read_csv('{}/data/data.csv.gz'.format(folder_source))
    clean_data = clean_data(df)
    print(' dataframe cleaned')
