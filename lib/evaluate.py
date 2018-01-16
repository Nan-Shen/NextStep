#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 05:40:10 2018

@author: Nan
"""
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from scikitplot.estimators import plot_learning_curve

################################
##       model evaluation     ##
################################     
def model_roc(sorted_models, selectdic, X_test, y_test, out_fp, name, top=3):
     """ROC plot of top models
     """
     plt.figure()    
     sns.set();sns.set_context({"figure.figsize": (20,20)});sns.set_context('talk')
     sns.set_style('white',{'font.family':'sans-serif',
                         'font.sans-serif':['Helvetica']})
     plt.xlim([-0.01, 1.00])
     plt.ylim([-0.01, 1.01])
     for i in range(top):
         params, model = sorted_models[i]
         #transform test data
         select, modelname = params
         X_test_fs = transform_test(select, selectdic, X_test)
         if modelname == 'SVM':
             y_score = model.decision_function(X_test_fs)
         elif modelname == 'NN':
             y_score = model.predict_proba(X_test_fs)[:,1]
         fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
         roc_auc = auc(fpr, tpr)
         precision = model.score(X_test_fs, y_test)
         plt.plot(fpr, tpr, lw=3, alpha=0.7, 
                  label='{:s}, {:s}(precision={:0.2f}, area={:0.2f})'.format(select, modelname, precision, roc_auc))
    
     plt.xlabel('False Positive Rate', fontsize=16)
     plt.ylabel('True Positive Rate (Recall)', fontsize=16)
     plt.plot([0, 1], [0, 1], color='k', lw=0.2, linestyle='--')
     plt.legend(loc='lower right', fontsize=11)
     plt.title('%s ROC curve' % name, fontsize=16)
     plt.axes().set_aspect('equal')
     plt.savefig('%s/%s_ROC.jpg' % (out_fp, name), bbox_inches='tight')
     
def model_learningcurve(model, params, Xdic, y_train, out_fp):
    """Classification plot of model
    """
    select, modelname = params
    X_train_fs = Xdic[select]
    name = 'Learning Curve(%s)' % (plot_label)
    plot_label = '%s.%s_classifier' % (select, modelname)
    plt.figure()    
    sns.set();sns.set_context({"figure.figsize": (20,20)});sns.set_context('talk')
    sns.set_style('white',{'font.family':'sans-serif',
                         'font.sans-serif':['Helvetica']})
    plot_learning_curve(model, X_train_fs, y_train, title=name, scoring='precision')
    plt.savefig('%s/%s_learning_curve.png' % (out_fp, plot_label), bbox_inches='tight')

def transform_test(select, selectdic, X_test):
    """Transform test features in the same way as training data
    """ 
    if '+' in select:
        fs, pca = select.split('+')
        X_test_fs = X_test[:, selectdic[fs]]
        X_test_fs = selectdic[select].transform(X_test_fs)
    elif select != 'PCA':
        X_test_fs = X_test[:, selectdic[select]]
    else:
        X_test_fs = selectdic[select].transform(X_test)
    return X_test_fs