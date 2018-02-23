import numpy as np
import pandas as pd
import sys
import csv

cutoffs = [7, 9, 10, 12, 15, 17, 20, 24, 29, 38, 56, 2009]

def extractPDs(inputfile, train, dev, test):
	df = pd.read_csv(inputfile, sep = '\t', header = 0)
	masterLength = len(df['price'])
	df['price-bucket'] = pd.Series(np.zeros(masterLength), index = df.index)
	for i in range (0, masterLength):
		df.at[i, 'price-bucket'] = getBucket(df['price'][i])
	df = df.sample(frac=1).reset_index(drop=True)
	traindf = df.truncate(before = 0, after= 1452884)
	devdf = df.truncate(before = 1452885, after= 1467709)
	testdf = df.truncate(before = 1467710, after = 1482534)
	print ("Size of training dataframe is " + str(traindf.shape))
	print ("Size of dev dataframe is " + str(devdf.shape))
	print ("Size of test dataframe is " + str(testdf.shape))
	print ("Size of master dataframe is " + str(df.shape))
	traindf.to_csv(train, index = False)
	devdf.to_csv(dev, index = False)
	testdf.to_csv(test, index = False)

def getBucket(price):
	for i in range(0, len(cutoffs)):
		if price < cutoffs[i]:
			return int(i)
	return int(len(cutoffs) - 1)

def main():
	np.random.seed(seed=1)
	if len(sys.argv) != 5:
		raise Exception("usage: python extractPDs.py <infile>.tsv <train>.csv <dev>.tsv <test>.tsv")
	tsvFilename, trainCSV, devCSV, testCSV = sys.argv[1:5]
	if tsvFilename == 'master.tsv':
		extractPDs(tsvFilename, trainCSV, devCSV, testCSV)
	else:
		print('Off by 1 error')

if __name__ == '__main__':
	main()