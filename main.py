
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
from sklearn.preprocessing import LabelEncoder
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
    datasetPath = './data/file.csv'
    df = pd.read_csv(datasetPath, sep=',') # separated by comma

    # extract header values
    tweets_data = df['tweets'].values
    labels_data = df['labels'].values
    # encode labels ["neutral, good, bad"] to [2, 1, 0]
    encoded_labels = LabelEncoder().fit_transform(labels_data)
    print(encoded_labels)
    print("Number of tweets: ", tweets_data.size)
    print("Number of labels: ", labels_data.size)
    print("Classes: ", ['neutral', 'good', 'bad'])
    print("")

    # process text samples
    tweets_cleaned_data = [clean_text(tweet) for tweet in tweets_data] # clean tweets data
   
    MAX_NUM_WORDS = 210000 # 210,000
    MAX_SEQUENCE_LENGTH = 50
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(tweets_cleaned_data)
    sequences = tokenizer.texts_to_sequences(tweets_cleaned_data)
    word_index = tokenizer.word_index # the dictionary
    print('Found %s unique tokens.' % len(word_index)) #only top MAX_NUM_WORDS will be used to generate the sequences
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of samples:', data.shape)
    print('Sample:(the zeros at the begining are for padding text to max length)')
    print(data[10])
    print(len(data[10]))
    print("")

    # format output of the CNN (the shape of labels)
    labels_matrix = to_categorical(np.asarray(encoded_labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels_matrix.shape)
    print('Sample label:\n', labels_matrix[2000])
    print("")

    # split samples and labels to training and testing sets
    TESTING_SPLIT = 0.2
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data_shuffled = data[indices]
    labels_shuffled = labels_matrix[indices]
    nb_validation_samples = int(TESTING_SPLIT * data_shuffled.shape[0])
    x_train = data_shuffled[:-nb_validation_samples]
    y_train = labels_shuffled[:-nb_validation_samples]
    x_val = data_shuffled[-nb_validation_samples:]
    y_val = labels_shuffled[-nb_validation_samples:]
    print('Shape of training data: ', x_train.shape)
    print('Shape of testing data: ', x_val.shape)
    print("")

    # read glove word embeddings
    EMBEDDING_DIM = 100
    print('Indexing word vectors.')
    embeddings_index = {}
    with open('data/glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split(sep=' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))

    # Map the dataset dictionary of (words,IDs) to a matrix of the
    # embeddings of each word in the dictionary
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM)) # +1 to include the zeros vector for non-existing words
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print ('Shape of Embedding Matrix: ', embedding_matrix.shape)

    # Build the deep NN
    embedding_layer = Embedding(len(word_index) + 1, #vocab size
    EMBEDDING_DIM, #embedding vector size
    weights=[embedding_matrix], #weights matrix
    input_length=MAX_SEQUENCE_LENGTH, #padded sequence length
    trainable=False)


    



    
# RUNNER
if __name__ == "__main__":
    main()