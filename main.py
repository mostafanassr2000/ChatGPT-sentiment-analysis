import numpy as np
from keras.layers import Embedding
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.models import Model

from DatasetHandler import DatasetHandler

def main():
    # read data from given dataset
    datasetPath = './data/file.csv'
    MAX_NUM_WORDS = 210000  # 210,000 or 6500
    MAX_SEQUENCE_LENGTH = 1000
    TESTING_SPLIT = 0.2
    # read, clean and tokenize data
    datasetHandler = DatasetHandler(
        datasetPath, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, TESTING_SPLIT, applyLemma=False)
    # get splitted data for CNN model
    dictionary = datasetHandler.getDictionary()
    encodedLabels = datasetHandler.getEncodedLabels()

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
    # +1 to include the zeros vector for non-existing words
    embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
    for word, i in dictionary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('Shape of Embedding Matrix: ', embedding_matrix.shape)

    # Build the deep NN
    embedding_layer = Embedding(
        len(dictionary) + 1,  # vocab size
        EMBEDDING_DIM,  # embedding vector size
        weights=[embedding_matrix],  # weights matrix
        input_length=MAX_SEQUENCE_LENGTH,  # padded sequence length
        trainable=False # do not update weights during training
    )

    # Build 1D CNN layers
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x) # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)
    print("cool")

    # split data to training and testing
    x_train, x_val, y_train, y_val = datasetHandler.splitSamples()
    # Build, Compile, and Run the model
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # happy learning!
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
    epochs=5, batch_size=128)

    # Evaluate the model
    print('Acuracy on testing set:')
    model.evaluate(x_val,y_val)

    # Prediction
    model.predict(x_val)


    """
    sample = 1
    label_vec = model.predict(data[sample].reshape(1,-1))
    label_id = np.argmax(label_vec)
    label_name = ''
    for name, ID in labels_index.items(): # for name, age in dictionary.iteritems(): (f
    or Python 2.x)
    if label_id == ID:
    label_name = name
    break
    print ('The category of article no %s is %s' %(sample ,label_name))
    """


# RUNNER
if __name__ == "__main__":
    main()
