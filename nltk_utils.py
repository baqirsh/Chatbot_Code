import nltk
import numpy as np
#nltk.download('punkt')
""" import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download() """
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenize_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenize_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

sentence = ["hello", "how", "are","you"]
words = ["hi","hello","i","you", "bye","thank", "cool"]
bag = bag_of_words(sentence, words)
print(bag)