import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download the WordNet corpus
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class Lemmatizer:
    lemmatizer = WordNetLemmatizer()
    def __init__(self):
       pass

    def lemmatize(self, sentence):
        # tokenize the sentence into words
        words = word_tokenize(sentence)

        # tag the words with their part-of-speech
        tagged_words = nltk.pos_tag(words)

        # lemmatize each word based on its part-of-speech tag
        lemmatized_words = []
        for word, tag in tagged_words:
            pos = self.getWordPos(tag)
            if pos:
                lemmatized_words.append(self.lemmatizer.lemmatize(word, pos=pos))
            else:
                lemmatized_words.append(self.lemmatizer.lemmatize(word))

        lemmatized_sentence = ' '.join(lemmatized_words)
        return lemmatized_sentence

    # Function to map the part-of-speech tags to WordNet tags
    def getWordPos(self, tag=''):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


# Example sentences
sample = [
    "The cats are playing in the garden.",
    "She is running to catch the bus.",
    "The books on the shelf are organized.",
    "He cooked a delicious meal for his family.",
    "I saw a beautiful sunset at the beach.",
    "The students are studying for their exams.",
    "The birds are chirping in the trees.",
    "She wore a stunning red dress to the party.",
    "He plays the guitar with great skill.",
    "They built a new house in the neighborhood."
]

