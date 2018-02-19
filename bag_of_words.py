from collections import Counter
import numpy as np
import pandas as pd
import re
import sys


def main():
    if len(sys.argv) != 2:
        raise Exception("usage: python train.csv")
    inputfile = sys.argv[1]
    df = pd.read_csv(inputfile, header = 0)
    descriptions = df['item_description'].as_matrix()
    count = 0
    vocabulary = dict()
    masterLength = len(descriptions)
    for i in range (0, masterLength):
        if (pd.isnull(descriptions[i]) == False):
            vocabulary, count = build_vocabulary(descriptions[i], count, vocabulary)
    encodings = []
    for i in range (0, len(descriptions)):
        if (pd.isnull(descriptions[i]) == False):
            encodings.append(create_document_vector(sentenceToArr(descriptions[i]), vocabulary))
        else:
            print "At iteration ", i, ", no product description was available"
            encodings.append((np.zeros((len(vocabulary)))))
    df['encodings'] = pd.Series(encodings, index = df.index)
    # print df['encodings']
    df.to_csv('train_enc.csv', index = False)

def build_vocabulary(sentence, Count, vocabulary):
    # For each word in words vector put in dictionary with index
    words= sentenceToArr(sentence)
    count = Count
    for word in words:
        if word in vocabulary:
            count = count
        else:
            count = count + 1
            vocabulary[word] = count
    return vocabulary, count

def sentenceToArr(sentence):
    return re.split(' |; |, |\*|\n', sentence)

def sentence_count(sentence):
    # Count occurance of each word in sentence
    word_count = dict(Counter(sentence))
    return word_count

def create_document_vector(description, vocabulary):
    # Build vector for sentence coding, at index in vocabulary put word frequency
    word_count = sentence_count(description)
    encode = np.zeros((len(vocabulary)))
    for word, count in word_count.iteritems():
        if word in vocabulary:
            index =  vocabulary[word]
            encode[index - 1] = count
    return encode




if __name__ == "__main__":
    main()