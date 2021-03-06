#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 03:40:23 2018

@author: Nan
"""

from __future__ import division

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

################################
##        preprocess          ##
################################
def preprocess(data, n, stdize=True):
    """preprocess data, including add rise column, split traininf and 
    testing data sets, standardize feature values.
    """
    prep_data, y = clean(data, n)
    X_train, X_test, y_train, y_test = split_traintest(prep_data, y, n, test_size=100)
    if stdize:
       X_train_scaled, X_test_scaled = standardize(X_train, X_test)
       return X_train_scaled, X_test_scaled, y_train, y_test 
    else:
       return X_train, X_test, y_train, y_test
        
def clean(data, n):
    """Remove dates with NAs in any column,
    Add nth working day from first day column(order of day exclude weekends)
    Add column indicate if price of n days later is higher or lower than current
    price.
    data:features of a stock, SP&500 and NASDAQ in the same time period.
    n: build model to predict if price rise or drop n days laster
    """
    data = data.sort_index()
    data['Order'] = range(data.shape[0])
    data = data.dropna(axis=0, how='any')
    prep_data = data.set_index(data['Order'])
    prep_data = prep_data.drop(['Order'], axis=1)
    y = pd.DataFrame()
    rise = ifrise(prep_data, n)
    filt = map(lambda i:i>=0, rise)
    y['Rise.'+str(n)] = rise
    y.index = prep_data.index
    y = y[filt]
    prep_data = prep_data[filt]
    return prep_data, y

def ifrise(prep_data, n):
    """Check if price of n days later is higher or lower than current price.
    n: compare day i to day i+n price
    """
    rise = []
    for i in prep_data.index:
        if i+n in prep_data.index:
           rise.append(int(prep_data['tg_Price'].loc[i+n] > prep_data['tg_Price'].loc[i]))
        else:
           rise.append(-1) 
    return rise

def split_traintest(prep_data, y, n, test_size=100):
    """Split train and test data.
    prep_data: preprocessed dataframe with features and if price rise column
    n: determine y value, compare day i with day i+n
    """
    #remove dates without y values
    filt = y['Rise.'+str(n)].isnull()
    X = prep_data[~filt]
    y = y['Rise.'+str(n)][~filt]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=0)
    return X_train, X_test, y_train, y_test
    
def standardize(X_train, X_test):
    """standardize feature values.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

################################
##    features transform      ##
################################
def poly_feature(X, degree):
    """generate more features using existing features’ high-order and 
    interaction terms.
    X: original features
    degree: degree of the polynomial features.
    """
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X) 
    return X_poly

################################
##    features selection      ##
################################ 
def feature_selection(X, y, varRatio1=0.001, impQuantile1=25):
    """make a dicionary of all selection and combos.
    X:training data set features.
    y:training data set values to be predicted.
    varRatio1: variance ratio threshold for PCA only selection. defualt=0.001.
    varRatio2: variance ratio threshold for PCA in RF + PCA selection. defualt=0.001.
    impQuantile1: keep features with importance score above impQuantile percent
    quantile in RF only selection. default=25
    """
    Xdic = {}
    selectdic = {}
    #no feature selection
    Xdic['noFS'] = X
    selectdic['noFS'] = [True] *  X.shape[1]
    #PCA only
    Xdic['PCA'], selectdic['PCA'] = pca_reduct(X)
    #RF only
    Xdic['RF'], selectdic['RF'] = rf_select(X, y, impQuantile=impQuantile1)
    #RF + PCA
    Xdic['RF+PCA'], selectdic['RF+PCA'] = pca_reduct(Xdic['RF'])
    #RFE only
    estimator = LogisticRegression() 
    #NN, svc does not provide feature importance information, cannot be estimator
    Xdic['RFE'], selectdic['RFE'] = rfe_select(estimator, X, y, score='precision')
    #RFE + PCA
    Xdic['RFE+PCA'], selectdic['RFE+PCA'] = pca_reduct(Xdic['RFE'])
    return Xdic, selectdic
  
def pca_reduct(X):
    """Linear dimensionality reduction using Singular Value Decomposition of 
    the data to project it to a lower dimensional space.
    X: features
    varRatio: variance ratio threshold, singular values explained more than 
    this threshold will be kept
    """
    pca = PCA(random_state=0)
    pca.fit(X)
    X_pca = pca.transform(X)
    return X_pca, pca
 
def rfe_select(estimator, X, y, score='precision'):
    """remove one important features recursively, then rank features by importance.
    estimator: model used to estimate, best number of features to retain
    score: selection critea, deault is precision. 
    """
    selector = RFECV(estimator, step=1, cv=5, scoring=score)
    selector = selector.fit(X, y)
    X_select = X[:,selector.support_]
    return X_select, selector.support_
    
def rf_select(X, y, impQuantile=25):
    """use ensembles of decision trees to calculate feature inportance.
    impQuantile: keep features with importance score above impQuantile percent
    quantile. default=25
    """    
    estimator = ExtraTreesClassifier(random_state=0)
    estimator.fit(X, y)
    minImp = np.percentile(estimator.feature_importances_, impQuantile)
    selector = map(lambda s:s > minImp, estimator.feature_importances_ )
    X_select = X[:,selector]
    return X_select, selector

################################
##       model selection      ##
################################
def model_selection(y, Xdic):
    """Test differnt combinations of feature selestions and models
    """
    model_dic = {}
    for x in Xdic:
        model_dic[(x, 'SVM')] = svm_model(Xdic[x], y)
        
        model_dic[(x, 'NN')] = nn_model(Xdic[x], y)
        
    sort_model = sorted(model_dic.items(), 
                        key=lambda x:x[1].best_score_, 
                        reverse=True)
    return sort_model

def svm_model(X_train, y_train, score='precision', gammas=[0.1, 1, 10, 100, 1000]):
    """select the best parameters for svm model.
    """
    svm = SVC(kernel='rbf')
    grid_values = {'gamma': gammas}
    
    grid_svm = GridSearchCV(svm, param_grid=grid_values, scoring=score)
    grid_svm.fit(X_train, y_train)
    return grid_svm

def nn_model(X_train, y_train, score='precision', alphas=[0.01, 0.1, 1, 5, 10],
                                                  layers=[1, 10, 100, [5, 5], [10, 10]]):
    """select the best parameters for neural network.
    """
    nn = MLPClassifier(solver='lbfgs', random_state = 0)
    grid_values = {'activation': ['relu', 'logistic', 'tanh'],
                   'alpha': alphas,
                   'hidden_layer_sizes':layers}
    
    grid_nn = GridSearchCV(nn, param_grid=grid_values, scoring=score)
    grid_nn.fit(X_train, y_train)
    return grid_nn
