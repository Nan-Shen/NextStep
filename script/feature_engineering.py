#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:57:14 2018

@author: Nan
"""

from stockPred.parse import parse_input, start_end, add_time_order



def read_input(in_fp, startDate, endDate):
    """
    """
    data = parse_input(in_fp)
    se_data = start_end(data, startDate, endDate)
    ord_data = add_time_order(se_data)
    
    