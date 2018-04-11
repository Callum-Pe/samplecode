from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import re
import re

#read in csv
table = pd.read_csv("ra_data_classifier.csv",encoding='latin1')
table_x = table.iloc[:,1:2]
table_y = table.iloc[:,2]
X = table_x.values.ravel()
Y = table_y.values

#remove junk characters 
regex = re.compile('[^$.,@\w\s\-_]|[\n]|[\r]|[\t]')
Z = X
i=0
for x in X:
    Z[i] = regex.sub("",x)
    i = i+1
X=Z



"""
Leave one out validation. This takes a long time to run and has minimal effect on accuracy. I wrote and ran it anyway because I had a fast machine.
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
bnb = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', BernoulliNB())])
sm = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', svm.SVC(C=5000))])
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
bst = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', ensemble.GradientBoostingClassifier(**params))])
mlp = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(40, 40), random_state=1))])
loo = LeaveOneOut()
bnb_score = 0
sm_score = 0
bst_score = 0
mlp_score=0
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    bnb.fit(X_train,y_train)
    bnb_score = bnb_score + accuracy_score(y_test, bnb.predict(X_test))
    sm.fit(X_train,y_train)
    sm_score = sm_score + accuracy_score(y_test, sm.predict(X_test))
    bst.fit(X_train,y_train)
    bst_score = bst_score + accuracy_score(y_test, bst.predict(X_test))
    mlp.fit(X_train,y_train)
    mlp_score = mlp_score + accuracy_score(y_test, mlp.predict(X_test))
print(bnb_score/100)
print(sm_score/100)
print(bst_score/100)
print(mlp_score/100)
"""


#compares classifiers. This method does 35 fold validation and then averages the scores over the n folds
# This method also does implicitly split data into train and tests sets, so it shouldnt be overfit.

folds = 35
bnb = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', BernoulliNB())])
print("bnb scores: {}".format(np.mean(cross_val_score(bnb, X, Y, cv=folds))))
sm = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', svm.SVC(C=5000))])
print("svm scores: {}".format(np.mean(cross_val_score(sm, X, Y, cv=folds))))
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
bst = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', ensemble.GradientBoostingClassifier(**params))])
print("bst scores: {}".format(np.mean(cross_val_score(bst, X, Y, cv=folds))))
mlp = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(40, 40), random_state=1))])
print("mlp scores: {}".format(np.mean(cross_val_score(mlp, X, Y, cv=folds))))


# This produces final classifier. 
final = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', svm.SVC(C=5000))])
final.fit(X,Y)
final_predictions = final.predict(X)
#This svm is perfectly fit to train set. This might be an sign of overfitting.
if Y.all() == final_predictions.all():
    print("exact")

