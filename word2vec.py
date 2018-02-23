from collections import Counter
import numpy as np
import pandas as pd
import re
import sys
import nltk

w2v = None
with open("glove.6B.100d.txt", "rb") as lines:
	w2v = {line.split()[0]: np.array(line.split()[1:])
    	   	for line in lines}

#dim = len(word2vec.itervalues().next()) - use this if running Python2!!!
dim = 100

def mean_vector(description):
	words = nltk.word_tokenize(description)
	result = np.zeros(dim)
	n = len(words)
	for w in words:
		if w.encode('UTF-8') in w2v:
			add = w2v[w.encode('UTF-8')]
		else: 
			add = np.zeros(dim)
		result += np.float64(add)
	result /= n
	return result

def main():

	if len(sys.argv) != 4:
        	raise Exception("usage: python bag_of_words.py train.csv dev.csv test.csv")
        
	inputfile = sys.argv[1]
	devfile = sys.argv[2]
	testfile = sys.argv[3]
	
	files = [inputfile, devfile, testfile]

	for fileName in files:
		df = None
		df = pd.read_csv(fileName, header = 0)
		descriptions = df['item_description'].as_matrix()
		count = 0
		vocabulary = dict()
		masterLength = len(descriptions)
		lengths = [int(masterLength/8), int(masterLength/4), int(3*masterLength/8), int(masterLength/2),int(5*masterLength/8), int(3*masterLength/4), int(7*masterLength/8)]
		for i in range (0, masterLength):
			if (pd.isnull(descriptions[i]) == False):
				vocabulary, count = build_vocabulary(descriptions[i].lower(), count, vocabulary)
		vocabulary['UNK'] = count
		writeVocabLengthToFile(len(vocabulary))
		encodings = []
		w2v_vectors = []
		j = 0
		for i in range (0, len(descriptions)):
			print("iteration:", i)
			if j!=7:
				if i == lengths[j]:
					print ("Currently ", j+1, "/8 of the way through encoding")
					j = j+1
			if (pd.isnull(descriptions[i]) == False):
				encodings.append(create_document_vector(sentenceToArr(descriptions[i].lower()), vocabulary))
#					print(mean_vector(descriptions[i]))
				w2v_vectors.append(mean_vector(descriptions[i]))
			else:
				print ("At iteration ", i, ", no product description was available")
				encodings.append(np.array([]))
				w2v_vectors.append(np.array([]))
		df['encodings'] = pd.Series(encodings, index = df.index)
		df['word2vec'] = pd.Series(w2v_vectors, index = df.index)
		# print df['encodings']
		enc_csv_name = fileName.split(".")[0] + "_enc.csv"
		df.to_csv(enc_csv_name, index = False)

def writeVocabLengthToFile(vocabLength):
    fp = open('vocab-length.txt', 'w')
    fp.write('Length of vocabulary is ' + str(vocabLength) +'.')
    fp.close()

def build_vocabulary(sentence, Count, vocabulary):
    # For each word in words vector put in dictionary with index
    words= sentenceToArr(sentence)
    count = Count
    for word in words:
        if word in vocabulary:
            count = count
        else:
            vocabulary[word] = count
            count = count + 1
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
    encode = []
    for word, count in list(word_count.items()):
        if word in vocabulary:
            index =  vocabulary[word]
            encode.append(index)
    return np.array(encode)

if __name__ == "__main__":
    main()
