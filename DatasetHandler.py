
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from Lemmatizer import Lemmatizer
import re
import numpy as np

class DatasetHandler:
    tweets_data = []
    labels_data = []
    encoded_labels = []
    lemmatizer = Lemmatizer()
    applyLemma = False

    tokenizedData = []
    dictionary = []

    MAX_NUM_WORDS = 0
    MAX_SEQUENCE_LENGTH = 0
    TESTING_SPLIT = 0

    def __init__(self, dataSetPath='./data/file.csv', MAX_NUM_WORDS=6500, MAX_SEQUENCE_LENGTH=50, TESTING_SPLIT=0.2, applyLemma=False):
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.TESTING_SPLIT = TESTING_SPLIT
        self.applyLemma = applyLemma
        # read data
        self.readDataset(dataSetPath)
        # filter and clean data
        self.tokenizeData()

    def readDataset(self, path=''):
        dataframe = pd.read_csv(path, sep=',')  # separated by comma
        # extract header values
        tweets_data = dataframe['tweets'].values
        labels_data = dataframe['labels'].values
        # encode labels ["neutral, good, bad"] to [2, 1, 0]
        encoded_labels = LabelEncoder().fit_transform(labels_data)
        print(encoded_labels)
        print("Number of tweets: ", tweets_data.size)
        print("Number of labels: ", labels_data.size)
        print("Classes: ", ['neutral', 'good', 'bad'])
        print("")
        self.tweets_data = tweets_data
        self.encoded_labels = encoded_labels

    def tokenizeData(self):
        # clean tweets data
        self.tweets_data = [self.clean_text(tweet)
                            for tweet in self.tweets_data]
        # tokenization
        tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(self.tweets_data)
        sequences = tokenizer.texts_to_sequences(self.tweets_data)
        self.dictionary = tokenizer.word_index  # the dictionary
        # only top MAX_NUM_WORDS will be used to generate the sequences
        print('Found %s unique tokens.' % len(self.dictionary))
        # perform padding
        data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        print('Shape of samples:', data.shape)
        print('Sample:(the zeros at the begining are for padding text to max length)')
        print(data[10])
        print("")
        self.tokenizedData = data

    def splitSamples(self):
        labels_matrix = to_categorical(np.asarray(self.encoded_labels))
        print('Shape of data tensor:', self.tokenizedData.shape)
        print('Shape of label tensor:', labels_matrix.shape)
        print('Sample label:\n', labels_matrix[2000])
        print("")
        # split samples and labels to training and testing sets
        # Using a fixed value for random_state allows you to reproduce the same train-test split every time you run the code
        x_train, x_val, y_train, y_val = train_test_split(self.tokenizedData, labels_matrix, test_size=self.TESTING_SPLIT, random_state=42)
        print('Shape of training data: ', x_train.shape)
        print('Shape of testing data: ', x_val.shape)
        print("")
        return x_train, x_val, y_train, y_val

    ## Getter ##

    def getDictionary(self):
        return self.dictionary
    
    def getEncodedLabels(self):
        return self.encoded_labels

    ## Helper Functions ##
    def clean_text(self, text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'\@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.lower()  # Convert to lowercase
        if self.applyLemma:
            text = self.lemmatizer.lemmatize(text)  # Lemmatization
        return text
