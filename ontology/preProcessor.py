import codecs

import nltk
from nltk.corpus import stopwords


def cleaning():
    with codecs.open("E:/Projects/HelaNER/data/inputText.txt", encoding='utf-8') as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)
    stop_words = set(stopwords.words("sinhala"))
    phrases = []

    for line in sentences:
        words = nltk.word_tokenize(line)
        cleaned_text = " ".join(list(filter(lambda x: x not in stop_words, words)))
        phrases.append(cleaned_text)
    return phrases
