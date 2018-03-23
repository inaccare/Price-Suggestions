"""
Script builds new CSV files from train, dev and test sets to add condensed
array bag of words representation for each product description. Each condensed
array has indices for non-zero entries in the corpus.
"""


from collections import Counter
import numpy as np
import pandas as pd
import re
import sys


def main():
    if len(sys.argv) != 4:
        raise Exception("usage: python bag_of_words.py train.csv dev.csv test.csv")
    inputfile = sys.argv[1]
    devfile = sys.argv[2]
    testfile = sys.argv[3]
    df = pd.read_csv(inputfile, header = 0)
    descriptions = df['item_description'].as_matrix()
    count = 0
    vocabulary = dict()
    masterLength = len(descriptions)
    lengths = [int(masterLength/8), int(masterLength/4), int(3*masterLength/8), int(masterLength/2),int(5*masterLength/8), int(3*masterLength/4), int(7*masterLength/8)]
    # Building the vocabulary to use for encoding the sentences
    for i in range (0, masterLength):
        if (pd.isnull(descriptions[i]) == False):
            vocabulary, count = build_vocabulary(descriptions[i].lower(), count, vocabulary)
    vocabulary['UNK'] = count
    # Writes length of vocabulary to an output file so that this can be used in the model
    writeVocabLengthToFile(len(vocabulary))
    # Rest of main encodes training, dev and test sets using vocabulary derived from train set
    encodings = []
    j = 0
    for i in range (0, masterLength):
        if j!=7:
            if i == lengths[j]:
                print ("Currently ", j+1, "/8 of the way through encoding")
                j = j+1
        if (pd.isnull(descriptions[i]) == False):
            encodings.append(create_document_vector(sentenceToArr(descriptions[i].lower()), vocabulary))
        else:
            print ("At iteration ", i, ", no product description was available")
            encodings.append(np.array([]))
    print ('Training product descriptions encoded...')
    df['encodings'] = pd.Series(encodings, index = df.index)
    df.to_csv('train_boW.csv', index = False)
    print ('Training encodings written to csv ...')
    print ('Reading in dev and test sets into dataframe ...')
    devdf = pd.read_csv(devfile, header = 0)
    testdf = pd.read_csv(testfile, header = 0)
    print ('Starting encoding for dev set...')
    dev_descriptions = devdf['item_description'].as_matrix()
    devLength = len(dev_descriptions)
    dev_encodings = []
    for i in range (devLength):
        if (pd.isnull(dev_descriptions[i]) == False):
            dev_encodings.append(create_document_vector(sentenceToArr(dev_descriptions[i].lower()), vocabulary))
        else:
            print ('At iteration ', i, ' of dev set, no product description was available')
            dev_encodings.append(np.array([]))
    print ('Dev product descriptions encoded...')
    devdf['encodings'] = pd.Series(dev_encodings, index = devdf.index)
    devdf.to_csv('dev_boW.csv', index = False)
    print ('Dev encodings written to csv ...')
    print ('Starting encoding for test set...')
    test_descriptions = testdf['item_description'].as_matrix()
    testLength = len(test_descriptions)
    test_encodings = []
    for i in range (testLength):
        if (pd.isnull(test_descriptions[i]) == False):
            test_encodings.append(create_document_vector(sentenceToArr(test_descriptions[i].lower()), vocabulary))
        else:
            print ('At iteration ', i, ' of test set, no product description was available')
            test_encodings.append(np.array([]))
    print ('Test product descriptions encoded...')
    testdf['encodings'] = pd.Series(test_encodings, index = testdf.index)
    testdf.to_csv('test_boW.csv', index = False)
    print ('Test encodings written to csv ...')

def writeVocabLengthToFile(vocabLength):
    """
    Records the vocabulary length for use in model
    Arguments:
    vocabLength -- length of the vocabulary (corpus)
    Returns:
    nothing
    """
    fp = open('vocab-length.txt', 'w')
    fp.write('Length of vocabulary is ' + str(vocabLength) +'.')
    fp.close()

def build_vocabulary(sentence, Count, vocabulary):
    """
    For each word in words vector put in dictionary with index
    Arguments:
    sentence -- sentence of words to be added to the vocabulary
    Count -- track index for storing new words in the vocabulary
    vocabulary -- dictionary of vocabulary 
    Returns:
    vocabulary -- updated vocabulary
    count -- updated index count
    """
    words = sentenceToArr(sentence)
    count = Count
    for word in words:
        if word in vocabulary:
            count = count
        else:
            vocabulary[word] = count
            count = count + 1
    return vocabulary, count

def sentenceToArr(sentence):
    """
    Splits sentence into array of words
    Arguments:
    sentence -- data file sentence 
    Returns:
    sentence -- array containing words in that sentence
    """
    return re.split(' |; |, |\*|\n', sentence)

def sentence_count(sentence):
    """
    Produce dictionary with key value pairs word and frequency respectively
    Arguments:
    sentence -- sentence of words in example
    Returns:
    word_count -- dictionary mapping words to their frequency within the sentence
    """
    word_count = dict(Counter(sentence))
    return word_count

def create_document_vector(description, vocabulary):
    """
    Build vector for sentence coding, at index in vocabulary put 1 (binary coding)    
    Arguments:
    description -- product description from example
    vocabulary -- vocabulary for all sentences in training set
    Returns:
    encoding -- encoded vector with 1 in the corresponding index of the word in the vocabulary
    """
    word_count = sentence_count(description)
    encode = []
    for word, count in list(word_count.items()):
        if word in vocabulary:
            index =  vocabulary[word]
            encode.append(index)
        else:
            encode.append(vocabulary['UNK'])
    return np.array(encode)


if __name__ == "__main__":
    main()
