#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:29:35 2018

@author: Nan
"""
import click

from NextStep.parse import parse_yahoo_to_nextstep
from NextStep.stats import ALL_STATS, all_features
from NextStep.classify import preprocess, feature_selection, model_selection
from NextStep.evaluate import model_roc, model_learningcurve

"""This script will take in yahoo history stock price of target stock, SP&500,
NASDAQ as input. Then, it extracts time-series features of history stock prices 
and based on these feature, predict future changes using machine learning models. 
This tool considers different feature engeering methods, including RF, RFE and 
PCA. And also different machine learning models, including Neural Network and
SVM. It will output a sorted list of models and their best parameters and test
them on testing data set.
"""

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='0.4')


@click.option('-i', '--stock_fp', required=True,
              type=click.Path(exists=True),
              help='file path of target stock history prices.')
@click.option('-s', '--sp500_fp', required=True,
              type=click.Path(exists=True),
              help='file path of sp&500 index history prices.')
@click.option('-q', '--nasdaq_fp', required=True,
              type=click.Path(exists=True),
              help='file path of NASDAQ index history prices.')
@click.option('-o', '--out_fp', required=True,
              type=click.Path(exists=False),
              help='Output file pathway')
@click.option('--start', required=True,
               help='start date to take in as input data. e.g. 2013-01-03')
@click.option('--end', required=True,
               help='end date to take in input data. e.g. 2017-12-29')
@click.option('-n', '--n', required=True, 
              help='To predict n days later if price will increase or decrease.')

def best_model(stock_fp, sp500_fp, nasdaq_fp, start, end, n, out_fp):
    """
    """
    data = read_input(stock_fp, sp500_fp, nasdaq_fp, start, end)
    X_train, X_test, y_train, y_test = preprocess_data(data, n, stdize=True)
    select_feature_model(X_train, X_test, y_train, y_test, out_fp)

def read_input(stock_fp, sp500_fp, nasdaq_fp, start, end):
    """Parse and extract all the features of target stock, Sp&500, NASDAQ.
    stock_fp: file path of target stock history prices.
    sp500_fp:file path of SP&500 index history prices.
    nasdaq_fp:file path of NASDAQ index history prices.
    start: start date.
    end: end date.
    """
    stock = parse_yahoo_to_nextstep(stock_fp, start, end)
    sp500 = parse_yahoo_to_nextstep(sp500_fp, start, end)
    nasdaq = parse_yahoo_to_nextstep(nasdaq_fp, start, end)
    
    data = all_features([sp500, nasdaq, stock], price_col='Close',
                         stock_name=['SP&500', 'NASDAQ', 'tg'], 
                         features='all')
    return data

def preprocess_data(data, n, stdize=True):
    """Clean data, remove NAs, compare day i with day i+n price to get 
    prediction targets, split train and test data sets.
    data: features of target stock, sp&500, nasdaq
    n: predict n days later rise or drop
    stdize: if standardize feature values. default is True.
    """
    X_train, X_test, y_train, y_test = preprocess(data, n, stdize)
    return X_train, X_test, y_train, y_test

def select_feature_model(X_train, X_test, y_train, y_test, out_fp):
    """Use random forest and recursive feature elimination to select features.
    And PCA to reduce feature dimensinons. Neural network and SVM classification 
    model were tested on each combinations of feature selection and reduction.
    At same time, best params of each model were selected based on given data
    set.
    """
    Xdic, selectdic = feature_selection(X_train, y_train, varRatio1=0.001, impQuantile1=25)
    sorted_models = model_selection(y_train, Xdic)
    
    model_roc(sorted_models, selectdic, X_test, y_test, out_fp, top=6)
    params, model = sorted_models[0]
    model_learningcurve(model, params, Xdic, y_train, out_fp)
    
if __name__ == "__main__":
    best_model() 