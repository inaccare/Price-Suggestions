from collections import Counter
import numpy as np
import pandas as pd
import re
import sys
import nltk
import codecs

w2v = None
gloveFile = sys.argv[4]
with open(gloveFile, "rb") as lines:
	w2v = {line.split()[0]: np.array(line.split()[1:])
    	   	for line in lines}

#dim = len(word2vec.itervalues().next()) - use this if running Python2!!!
dim = len(w2v['them'.encode('UTF-8')])
def main():
	if len(sys.argv) != 5:
        	raise Exception("usage: python build_na_w2v_avg.py train.csv dev.csv test.csv gloveFile")
    
	inputfile = sys.argv[1]
	devfile = sys.argv[2]
	testfile = sys.argv[3]
	
	files = [inputfile, devfile, testfile]

	for fileName in files:
		df = pd.read_csv(fileName, header = 0)
		descriptions = df['item_description'].as_matrix()
		count = 0
		vocabulary = dict()
		masterLength = len(descriptions)
		lengths = [int(masterLength/8), int(masterLength/4), int(3*masterLength/8), int(masterLength/2),int(5*masterLength/8), int(3*masterLength/4), int(7*masterLength/8)]
		w2v_vectors = []
		j = 0
		for i in range (0, len(descriptions)):
			if j!=7:
				if i == lengths[j]:
					print ("Currently ", j+1, "/8 of the way through encoding")
					j = j+1
			if (pd.isnull(descriptions[i]) == False):
				w2v_vectors.append(mean_vector(descriptions[i]))
			else:
				print ("At iteration ", i, ", no product description was available")
				w2v_vectors.append(np.zeros(dim))
		df['word2vec'] = pd.Series(w2v_vectors, index = df.index)
		enc_csv_name = str(fileName.split(".")[0]) + "_w2v.csv" # Change naming back as well
		df.to_csv(enc_csv_name, index = False)
	return

def mean_vector(description):
	words = nltk.word_tokenize(description.lower())
	result = np.zeros(dim)
	n = len(words)
	for w in words:
		if w.encode('UTF-8') in w2v:
			add = w2v[w.encode('UTF-8')]
		else: 
			add = np.zeros(dim)
		result += np.float64(add)
	result /= n
	return np.array(result)

if __name__ == "__main__":
    main()
