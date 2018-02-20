# Price-Suggestions

The data that we are using can be obtained at (https://www.kaggle.com/c/mercari-price-suggestion-challenge).  The files provided at that link contain over 1.4 million items along with their prices, descriptions, conditions, etc.

makeCSVs.py is used to split the data into train, dev, and test sets

bag_of_words.py allows us to put the data into a more useful form to feed into our neural network.  It currently grabs only the item description for each item and represents it as a bag of words (i.e. a dictionary containing a count of which words from the vocabulary are contained in the item description).

model.py takes in the output of bag_of_words.py and runs it through a simple neural network and outputs a softmax vector which indicates the likelihood of the item being in each of the price buckets we defined.

Next Steps:
- Finish model.py
- Change output to being a linear function
- Tune parameters
- Begin using word2vec and an RNN framework
