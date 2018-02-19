
from collections import Counter
import numpy as np
import pandas as pd
import re
import sys

def main():
	if len(sys.argv) != 2:
		raise Exception("usage: python <encoded>.csv")
	csvFile = sys.argv[1]
	df = pd.read_csv(csvFile, header = 0)
	for i in range(0, len(df['item_description'])):
		print i,df['encodings'][i]

if __name__ == '__main__':
	main()