from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def preprocess(reviews):
	cleanReviews = []	#creating a new list of cleaned words
	for i in reviews:			# enumerate is a very helpful function that allows for a reliable count of iterations
		cleanReviews.append(preprocessHelper(i))	# call out to a helper function which actually does the preprocessing
		#print(cleanReviews)
	return cleanReviews	
	
# preprocessing helper function to minimize 'nested loops' looks prettier
def preprocessHelper(review):		# createed because i ran out of memory when doing a double for loop 
	stopWords = set(stopwords.words('english'))	#english stopwords
	lemmatizer = WordNetLemmatizer()		#currently using wordnet lemmatizer may change to snowball porter later depending
	lem = ''							# storing lemmatized words here
	#removing html tags
	noHTML = re.compile('<.*?>')			# html uses <> removes anything inside the brackets1
	# removing numbers and anything else
	review = re.sub(noHTML,'',review) 		# replace the html tokens with nothing ''
	data = re.sub("[^a-zA-Z]", " ",review)	# ensures that only alphabetical characters are part of the dataset
	data = data.lower().split()				# set each word to lower case and split.
	for i in data:	# easier to get rid of stop words with all the words separated
		if i not in stopWords:		
			lem += str(lemmatizer.lemmatize(i)) + ' '		#creates a bag of words ensure that the type as str
	words = word_tokenize(lem)
	return lem

		
def main():
	# file opening. using with open to allow python to take care of most of the file io issues for me
	with open("movieTrainingData.txt","r",encoding="utf-8") as f1:	# using with open to let python take care of closing correctly and other nuances with file i/o
		trainSet = f1.readlines()	# reviews are separated by lines. using utf-8 lest there is an error with reading the data(normalize text data)
	with open("movieTestData.txt","r",encoding="utf-8") as f2:	# using with open to let python take care of closing correctly and other nuances with file i/o
		testSet = f2.readlines()		# reviews are separated by lines. using utf-8 lest there is an error with reading the data(normalize text data)
		
		
	sentiment = [line.split("\t",1)[0] for line in trainSet]	# split the training data by score positive 1 or negative -1 found on first character
	reviews = [line.split("\t",1)[1] for line in trainSet]		# get the reviews after the +/-1
	
	train = preprocess(reviews)	#preprocessing the training set data
	test = preprocess(testSet)		#preprocessing the testing set data
	#creating the tfidf matrix by using built in libraries
	
	vectorizer = TfidfVectorizer()		# creating a tfidf matrix using sklearn library

	trainMatrix = vectorizer.fit_transform(train)		# fit and transform the train matrix
	#print(vectorizer.vocabulary_)
	#print(trainMatrix)
	testMatrix = vectorizer.transform(test)		# only transforming the testMatrix, if fit and transform then cosine similarity won't work
	#print(trainMatrix)
	
	cosSim = cosine_similarity(testMatrix,trainMatrix)		# find cosine similarity done through sklearn library
	
	#print(sentiment)
	testSentiment = []
	k = 30			# testing and setting k not sure what to set it to
	
	# k nearest neighbor implementation. 
	for i in cosSim:			# for each 'row' in the cosine similarity matrix
		# sort up to kth position for faster and more accurate references
		rows = np.argpartition(-i,k)	 # sorts and obtains the nearest neighbors = k. view documentation of np.argpartition for more info
		# print(rows)
		#sorted and finds k nearest neighbor
		knn = rows[:k]			# review[0-k]
		#print(knn)
		#print(i)
		pos = 0		# number of positive reviews in data will be used as counter to see whether the test set is a positive or negative review
		neg = 0		# number of negative reviews in data
		for j in knn:
			#print(sentiment[j])
			if int(sentiment[j]) == 1:	# type check train set sentiment of neighbors and see if it equals pos or neg
				pos+=1				# iterate accordingly
			else:
				neg+=1
		# print(posRev,negRev)	# use as tester
		if pos > neg:	# should there be more positive than negative sentiments for that review, then we can say that the review is positive. should they be equal set as positive(subject to change)
			testSentiment.append('1')	# append a positive sentiment to the test list
		else:
			testSentiment.append('-1')		# else append a negative sentiment
	out = open('output.txt','w')		# file output in a .dat file 
	out.writelines("%s\n" % x for x in testSentiment)		# write a line of string of reviews for each x counter in test Sentiment
	
	out.close()
	print("complete")
if __name__ == "__main__":		# python's pseudo main function
    main()		# using for helper functions like preprocess and such