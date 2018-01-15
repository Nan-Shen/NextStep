#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 03:40:23 2018

@author: Nan
"""

from __future__ import division

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.neural_network import MLPClassifier

################################
##        preprocess          ##
################################
def preprocess(data, n, stdize=True):
    """preprocess data, including add rise column, split traininf and 
    testing data sets, standardize feature values.
    """
    prep_data, y = clean(data, ndays)
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
    y['Rise.'+str(n)] = rise
    y.index = prep_data.index
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
           rise.append(np.nan) 
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

def pca_reduct(X, varRatio=0.001):
    """Linear dimensionality reduction using Singular Value Decomposition of 
    the data to project it to a lower dimensional space.
    X: features
    varRatio: variance ratio threshold, singular values explained more than 
    this threshold will be kept
    """
    pca = PCA(random_state=0)
    pca.fit(X)
    idx = sum(map(lambda r:r > varRatio, pca.explained_variance_ratio_))
    X_pca = pca.components_[:, :idx]
    return X_pca
 
def rfe_select(estimator, X, y, score='precision'):
    """remove one important features recursively, then rank features by importance.
    estimator: model used to estimate, best number of features to retain
    score: selection critea, deault is precision. 
    """
    selector = RFECV(estimator, step=1, cv=5, scoring=score, random_state=0)
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
def svm_model(X_train, y_train, score='precision'):
    """select the best parameters for svm model.
    """
    svm = SVC(kernel='rbf')
    grid_values = {'gamma': [0.1, 1, 10, 100, 1000]}
    
    grid_svm = GridSearchCV(svm, param_grid=grid_values, scoring=score)
    grid_svm.fit(X_train, y_train)
    return grid_svm

def nn_model(X_train, y_train, score='precision'):
    """select the best parameters for neural network.
    """
    nn = MLPClassifier(solver='lbfgs', random_state = 0)
    grid_values = {'activation': ['relu', 'logistic', 'tanh'],
                   'alpha': [0.01, 0.1, 1, 5, 10],
                   'hidden_layer_sizes':[1, 10, 100, [5, 5], [10, 10]]}
    
    grid_nn = GridSearchCV(nn, param_grid=grid_values, scoring=score)
    grid_nn.fit(X_train, y_train)
    return grid_nn

  
def combo(X, y, 
          varRatio1=0.001, varRatio2=0.001, varRatio3=0.001, impQuantile1=25):
    """Test differnt combinations of feature selestions and models
    """
    Xdic = {}
    #PCA only
    Xdic['PCA'] = pca_reduct(X, varRatio=varRatio1)
    #RFE only
    Xdic['RFE'], selector_rfe = rfe_select(estimator, X, y, score='precision')
    #RF only
    Xdic['RF'], selector_rf = rf_select(X, y, impQuantile=impQuantile1)
    #RFE + PCA
    Xdic['RFE+PCA'] = pca_reduct(X_rfe, varRatio=varRatio2)
    #RF + PCA
    Xdic['RF+PCA'] = pca_reduct(X_rf, varRatio=varRatio3)
    
   
    model_dic = {}
    for x in Xdic:
        model_dic[(x, 'SVM')] = svm_model(Xdic[x], y)
        
        model_dic[(x, 'NeuralNetwork')] = nn_model(Xdic[x], y)
        
    sort_model = sorted(score_dic.items(), 
                        key=lambda x:x[1].best_score_, 
                        reverse=True)
    best_model = sort_model[0][0]
    return best_model
    
################################
##       model evaluation     ##
################################     
def test_model(model, X_test, y_test):
     """Test top models on test data
     """
     y_decision = model.decision_function(X_test)
     roc_auc_score(y_test, y_decision)
     precision_score(y_test, y_decision)
    
    SVM 
    neural network
    RBF-Kernelized SVM and Grid Search on parameters:
        linear, polynomial or radial basis function
        C 0.1 - 1000
    bayesian networks
    basic model:random walk


y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 
roc_auc_score(y_test, y_decision_fn_scores_auc)
