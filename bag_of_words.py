
from collections import Counter
import numpy as np
import re


def main():
    words = ["the is black dog the is black dog, cow", "Monkeys are blue and black"]
    count = 0
    vocabulary = dict()
    for i in range (0, len(words)):
        vocabulary, count = build_vocabulary(words[i], count, vocabulary)

    encoding = np.zeros((len(vocabulary), len(words)))
    for i in range (0, len(words)):
        encoding[:, i] = create_document_vector(sentenceToArr(words[i]), vocabulary)

    print"encoding: ", encoding
    print"dictionary: ", vocabulary

def build_vocabulary(sentence, Count, dictionary):
    # For each word in words vector put in dictionary with index
    words= sentenceToArr(sentence)
    count = Count
    for word in words:
        if word in dictionary:
            count = count
        else:
            count = count + 1
            dictionary[word] = count
    return dictionary, count

def sentenceToArr(sentence):
    return re.split(' |; |, |\*|\n',sentence)

def sentence_count(sentence):
    # Count occurance of each word in sentence
    word_count = dict(Counter(sentence))
    return word_count

def create_document_vector(description, vocabulary):
    # Build vector for sentence coding, at index in vocabulary put word frequency
    word_count = sentence_count(description)
    encode = np.zeros((len(vocabulary)))
    for word, index in vocabulary.items():
        if word in word_count:
            encode[index - 1] = word_count[word]
    return encode




if __name__ == "__main__":
    main()
