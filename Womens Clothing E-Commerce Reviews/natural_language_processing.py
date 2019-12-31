# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C://Users//sylve//Downloads//Machine learning//Dataset//Project datasets modified//Womens Clothing E-Commerce Reviews//Womens Clothing E-Commerce Reviews.csv',index_col=0)

dataset.head()

df1=dataset.dropna(subset=['Title'])
df1=df1.dropna(subset=['Review Text'])
df1=df1.dropna(subset=['Division Name'])

df1.reset_index(inplace=True)

df1.head()
df1['Review Text'].shape

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 19662):
    review = re.sub('[^a-zA-Z]', ' ', df1['Review Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df1.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy = (TP + TN) / (TP + TN + FP + FN)
#Precision = TP / (TP + FP)
#Recall = TP / (TP + FN)
#F1Score = 2 * Precision * Recall / (Precision + Recall)

accuracy = (522+1033)/(522+1033+2197+181)
accuracy

precision = 522/(522+2197)
precision

recall=522/(522+181)
recall

f1score = 2 * precision*recall/(precision+recall)
f1score
