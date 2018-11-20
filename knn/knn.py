# -*- coding:utf8 -*-

"""
================================
Nearest Neighbors Classification
================================

Sample usage of Nearest Neighbors classification.
It will plot the decision boundaries for each class.
"""

import numpy as np
import os, sys

from sklearn import neighbors, datasets
from sklearn.cross_validation import train_test_split

def loadData():
    iris = datasets.load_iris()
    X = iris.data[:,:2]
    Y = iris.target
    return X, Y

''' n_neighbors : Number of neighbors to use by default for kneighbors queries
    weights: uniform(default) or distance
    algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
'''
def train(n_neighbors, weights, algorithm, train_X, train_Y):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights, algorithm)
    clf.fit(train_X, train_Y)
    return clf

def predict(model, X):
    return model.predict(X.reshape(-1, 2))


def accuray(test_Y, predict_Y):
    if len(test_Y) != len(predict_Y):
        print 'predict data count is not equal to true data count!!'
        os._exit(-1)
    acc = .0; cnt = len(test_Y)
    for i in range(cnt):
        if (test_Y[i] == predict_Y[i]):
            acc += 1.
    acc /= cnt 
    return acc
    
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'The Param [n_neighbors] is NULL'
        os._exit(-1)

    n_neighbors = int(sys.argv[1])

    print 'Start read data'

    X, Y = loadData()
    y_distinct = set(Y)
    print 'Y has %d labels' % len(y_distinct)

    if len(X) != len(Y):
        print 'loadData Errors!'
        os._exit(-1)

    test_size = 0.33
    random_state = 42
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

    # train the model
    print '----------- begin to train --------------'
    #n_neighbors = 27
    weights = 'uniform'
    algorithm = 'kd_tree'
    knn_model = train(n_neighbors, weights, algorithm, train_X, train_Y)

    # predict
    print '------------ begin to test --------------'
    cnt = 0
    #print len(test_X)
    predict_Y = predict(knn_model, test_X)

    print 'acc = %f' % accuray(test_Y, predict_Y)

