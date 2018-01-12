#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:49:26 2018

@author: Nan
"""
import sys 

import pandas as pd
from datetime import datetime

def parse_input(in_fp, form='yahoo'):
    """read input csv file of historical stock price.
    in_fp: full file path of input csv file.
    data: dataframe of stock prices, with 'Date' column in date.time data type
    and all the other column in float.
    """
    data = pd.read_csv(in_fp)
    if form == 'yahoo':
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = data[['Open',
            'High', 'Low', 'Close', 
            'Adj Close', 'Volume']].astype(float)
    else:
        sys.stderr.write('Please check input file source and format. Default is'
                         'Yahoo format.')
    return data

def start_end(data, start='2013-01-03', end='2017-12-29'):
    """slice data from start date to end date.
    data: dataframe of stock prices, with 'Date' column in date.time data type
    and all the other column in float.
    start: start date(included), default is '2012-01-03'.
    end: end date(included), default is '2017-12-29'.
    """
    startd = datetime.strptime(start, '%Y-%m-%d')
    endd = datetime.strptime(end, '%Y-%m-%d')
    se_data = data[(data['Date'] >= startd) & (data['Date'] <= endd)]
    return se_data

#There are some irregular national holidays. Hard to caculate working days in this way.
def nth_weekday(firstday, day):
    """calculate, given day is the nth weekday after the first day.
    firstday: date (datetime format) of first day
    day: target day date (datetime format)
    """
    days_firstweek = 6 - firstday.isoweekday()
    days_lastweek = day.isoweekday()
    days_between = ((day - firstday).days + 1 
                              - days_firstweek 
                              - days_lastweek) // 7 * 5
    return days_between + days_firstweek + days_lastweek
    
def add_time_order(data):
    """add weekday column and nth working day from first day (order of day exclude
    weekends) column.
    data: dataframe of stock prices, with 'Date' column in date.time data type
    and all the other column in float.
    """
    data['Weekday'] = data['Date'].apply(datetime.isoweekday)
    data = data.sort_values(by=['Date'])
    data['Order'] = range(data.shape[0])
    return data
    