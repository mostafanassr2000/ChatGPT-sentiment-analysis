
#from Tokenizer import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.models import Model


# Tokenization and cleaning
def clean_text(text):
    #print(word_tokenize(text))
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    #tokens = word_tokenize(text)  # Tokenization
    #print(tokens)
    return text


def main():
    # read data from given dataset
    datasetPath = './dataset/file.csv'
    df = pd.read_csv(datasetPath, sep=',') # separated by comma

    # extract header values
    tweets_data = df['tweets'].values
    labels_data = df['labels'].values
    print("Number of tweets: ", tweets_data.size)
    print("Number of labels: ", labels_data.size)
    print("Classes: ", ['good', 'neutral', 'bad'])

    # process text samples
    tweets_cleaned_data = [clean_text(tweet) for tweet in tweets_data] # clean tweets data
   
    MAX_NUM_WORDS = 30000 # 200,000
    MAX_SEQUENCE_LENGTH = 3000 # 10000
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(tweets_cleaned_data)
    sequences = tokenizer.texts_to_sequences(tweets_cleaned_data)
    word_index = tokenizer.word_index # the dictionary
    print('Found %s unique tokens.' % len(word_index)) #only top MAX_NUM_WORDS will be used to generate the sequences
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of samples:', data.shape)
    print('Sample:(the zeros at the begining are for padding text to max length)')
    print(len(data[2]))
    



    
# RUNNER
if __name__ == "__main__":
    main()