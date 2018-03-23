from collections import Counter
import numpy as np
import pandas as pd
import re
import sys
import nltk
import codecs
import multiprocessing
import gensim

"""
This python script trains a w2v encoding scheme based on the corpus of our
product descriptions. Outputs necessary files to compute w2v in various
models.
"""


def clean_and_split_str(string):
    strip_special_chars = re.compile("[^A-Za-z]+")
    string = re.sub(strip_special_chars, " ", string)
    return string.strip().split()

def main():
	if len(sys.argv) > 3 or len(sys.argv) < 2:
		raise Exception("usage: python build_w2v_vector.py train.csv <num_dim>")

	num_dim = 100
	if len(sys.argv) == 3:
		num_dim = sys.argv[2]

	inputfile = sys.argv[1]
	raw_df = pd.read_csv(inputfile, header = 0)
	print("data loaded")

	raw_corpus = u"".join(raw_df['item_description'].astype(str) + " ")
	print("Raw Corpus contains {0:,} words".format(len(raw_corpus.split())))

	tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
	print("The punkt tokenizer is loaded")
	raw_sentences = tokenizer.tokenize(raw_corpus)
	print("We have {0:,} raw sentences".format(len(raw_sentences)))

	sentences = []
	for raw_sent in raw_sentences:
		if len(raw_sent) > 0:
			sentences.append(clean_and_split_str(raw_sent.lower()))
	print("We have {0:,} clean sentences".format(len(sentences)))

	token_count = sum([len(sentence) for sentence in sentences])
	print("The dataset corpus contains {0:,} tokens".format(token_count))

	#Dimensionality of the resulting word vectors
	num_features = num_dim
	#Minimum word count threshold
	min_word_count = 2
	#Number of threads to run in parallel
	num_workers = multiprocessing.cpu_count() 
	#Context window length
	context_size = 7
	#Seed for the RNG, to make the result reproducible
	seed = 1
	
	word2vec_model = gensim.models.word2vec.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers, 
    size=num_features, 
    min_count=min_word_count, 
    window=context_size)
	     
	word2vec_model.build_vocab(sentences=sentences)
	print("The vocabulary is built")
	print("Word2Vec vocabulary length: ", len(word2vec_model.wv.vocab))
	 
	#Start training the model
	word2vec_model.train(sentences=sentences, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)
	print("Training finished")

	word2vec_model.save("Mercari_item_des_trained.w2v")
	print("Model saved")

	w2v_model = gensim.models.word2vec.Word2Vec.load("Mercari_item_des_trained.w2v")
	print("Model loaded")

	counter = 0
	for word in w2v_model.wv.vocab:
		print("word: ", word)
		print(w2v_model.wv[word])
		counter += 1
		if counter > 10:
			break

	# df_des = df['item_description']
	# df_des.to_csv('pandas.txt', header=None, index=None, sep=' ', mode='a', escapechar=" ")
	return

if __name__ == "__main__":
	main()