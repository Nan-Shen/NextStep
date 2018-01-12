#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:47:51 2018

@author: Nan
"""
from __future__ import division

import numpy as np
from sklearn import linear_model



########################################################
##    features of stockToPredict, NASDAQ, S&P 500  ##
########################################################



##############################
##  static stock indicators  ##
##############################
def prices(row):
    """Return adjusted closing price, Volume on each day.
    row: a row in input data, containing values above of a certain day.
    """
    price, close, high, low, volume = row[['Open', 'Close', 'High', 
                                           'Low', 'Volume']]
    return price, close, high, low, volume

def weekday():
    """Which weekday is the predicted day.
    Prices on Friday tend to rise. Prices on Monday has the least tendency to 
    rise. 
    Frank Cross. The behavior of stock prices on Fridays and Mondays. 
    Financial Analyst Journal Vol. 29 No. 6, pages 67-69.
    """
    

##############################
##  time series indicators  ##
##############################
"""A stock technical indicator is a series of data points that are derived by 
applying a function to the price data at time t and study period n.
"""
#price change
def rocr(data, n):
    """Rate of Change: Compute rate of change relative to previous trading 
    intervals.
    (Price(t)/Price(t-n))*100
    data: input dataframe
    n: study period
    return a list of ROCR (may contain NA)
    """
    return data['Open'] / data['Open'].shift(n) * 100
    
def mom(data, n):
    """Momentum: Measures the change in price.
    Price(t) - Price(t-n)
    """
    return data['Close'] - data['Close'].shift(n)

#buy & sell signals
def willr(data, n):
    """Williams %R: Determine where today’s closing price fell within the
    range of past 10-day’s transaction.
    (highest - Close(t))/(highest - lowest)*100
    """
    highest = data['High'].rolling(window=10).max()
    lowest = data['Low'].rolling(window=10).min()
    return (highest - data['Close']) / (highest - lowest) * 100
        
def rsi(data, n):
    """Relative Strength Index: Suggests the overbought and oversold market 
    signal.
    Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100
    Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)-Price(t-1)>0};
           PriceDown(t)=1*(Price(t-1)-Price(t)){Price(t)-Price(t-1)<0};
    n: study period
    """
    deltaPrice = data['Close'] - data['Close'].shift(1)
    priceUp = [d if d > 0 else np.nan for d in deltaPrice ]
    priceDown = [abs(d) if d < 0 else np.nan for d in deltaPrice ]
    avgUp = [np.nan] * n + [np.nanmean(priceUp[i-n+1: i+1]) for i in range(n, len(priceUp))]
    avgDown = [np.nan] * n + [np.nanmean(priceDown[i-n+1: i+1]) for i in range(n, len(priceDown))]
    return [u/(u + d) for u,d in zip(avgUp, avgDown)]
    

def cci(data, n):
    """Commodity Channel Index: Identifies cyclical turns in stock price.
    Tp(t)-TpAvg(t,n)/(0.15*MD(t)) 
    where: Tp(t)=(High(t)+Low(t)+Close(t))/3;
           TpAvg(t,n)=Avg(Tp(t)) over [t, t-1, …, t-n+1];
           MD(t)=Avg(Abs(Tp(t)-TpAvg(t,n)));
    n: study period
    """
    Tp = (data['High'] + data['Low'] + data['Close']) / 3
    TpAvg = Tp.rolling(n).mean()
    MD = abs(Tp - TpAvg).rolling(n).mean()
    return Tp - TpAvg/(0.15*MD)

def sma(price, n):
    """Simple Moving Average: average closing price of stock over last n periods.
    price: a list of prices
    n: study period
    """
    return [np.nan] * (n-1) + [np.mean(price[i-n+1: i+1]) for i in range(n-1, len(priceDown))]

def ema(price, data, n, Multiplier=False):
    """Exponential moving averages (EMAs) reduce the lag by applying more 
    weight to recent prices. There are three steps to calculating an 
    exponential moving average (EMA). First, calculate the simple moving 
    average for the initial EMA value. A simple moving average is used as the 
    previous period's EMA in the first calculation. Second, calculate the 
    weighting multiplier. Third, calculate the exponential moving average 
    for each day between the initial EMA value and today, using the price, 
    the multiplier, and the previous period's EMA value. 
    EMA = {Close - EMA(t-1)} x multiplier + EMA(t-1)
    where: Multiplier = (2 / (n + 1))
    price: a list of prices
    n: study period
    Multiplier: alpha, default is 2/(n+1) for EMA. It can be used to calculate
    modified moving average (MMA), running moving average (RMA), 
    or smoothed moving average (SMMA) when set to 1/n.
    """
    if Multiplier == False:
        Multiplier = (2.0 / (n + 1))
    ema_init = sma(price, n)
    for i in range(1, len(ema_init)):
        if not np.isnan(ema_init[i-1]):
           ema_init[i] = (data['Close'].iloc[i] - ema_init[i-1]) * Multiplier + ema_init[i-1]
    return ema_init
      
def macd(data):
    """Moving Average Convergence Divergenece: Use different EMA signal buy & 
    sell.
    OSC(t)-EMAosc(t) 
    where OSC(t)=EMA1(t)-EMA2(t); 
          EMAosc(t)=EMAosc(t-1)+(k*OSC(t)-EMAosc(t-1))
    price: a list of prices
    return MACD line(12day EMA - 26dayEMA), 
           signal line(9day EMA), 
           MACD Histogram (MACD line - Signal line)
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    """
    price = data['Close']
    outMACD = [i - j for i,j in zip(ema(price, data, 12), ema(price, data, 26))]
    outMACDSignal = ema(price, data, 9)
    outMACDHist = [i - j for i,j in zip(outMACD, outMACDSignal)]
    return outMACD, outMACDSignal, outMACDHist 
    
   
#stock trend discovery
def adx(data, n):
    """Avergae Directional Index: Discover if trend is developing.
    Sum((+DI-(-DI))/(+DI+(-DI))/n
    https://en.wikipedia.org/wiki/Average_directional_movement_index
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    """
    UpMove = data['High'] - data['High'].shift(1)
    DownMove = data['Low'].shift(1) - data['Low']
    
    plusDM = [u if u > d and u > 0 else 0 for u,d in zip(UpMove, DownMove)]
    minusDM = [d if u < d and d > 0 else 0 for u,d in zip(UpMove, DownMove)]
    
    plusDI = [100*i/j for i,j in zip(ema(plusDM, data, n, Multiplier=1/n), atr(data, n))]
    minusDI = [100*i/j for i,j in zip(ema(minusDM, data, n, Multiplier=1/n), atr(data, n))]
    DX = [100*abs(i-j)/(i+j) for i,j in zip(plusDI, minusDI)]
    return ema(DX, data, n, Multiplier=1/n)
    
def mfi(data, n):
    """Money Flow Index: Relates typical price with Volumn. 
    100-(100/(1+Money Ratio)) 
    where Money Ratio=(+Moneyflow/-Moneyflow);
                Moneyflow=Tp*Volume
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi
    """
    Tp = (data['High'] + data['Low'] + data['Close']) / 3
    Moneyflow = Tp * data['Volume']
    plusMoneyflow = (Moneyflow >= Moneyflow.shift(1)) * Moneyflow
    minusMoneyflow = (Moneyflow < Moneyflow.shift(1)) * Moneyflow
    Moneyratio = plusMoneyflow.rolling(n).sum() / minusMoneyflow.rolling(n).sum()
    return 100 - (100/(1 + Moneyratio))
    
def tsf(data):
    """Time Series Forcasting: Calculates the linear regression of 20-day price.
    Linear Regression Estimate with 20-day price.
    """
    t = range(20)
    t_var = np.var(t, ddof=1)
    rolling_cov = data['Close'].rolling(20, 20).apply(lambda x:np.cov(x, t, ddof=1)[1, 0])
    return rolling_cov / t_var    

#noise elimination and data smoothing    
def trix(data, n):
    """Triple Exponential Moving Average: Smooth the insignificant movements.
    TR(t)/TR(t-1) 
    where TR(t)=EMA(EMA(EMA(Price(t)))) over n days period
    price: a list of prices
    n: study period
    """
    price = data['Close']
    TR = ema(ema(ema(price, data, n), data, n), data, n)
    return [np.nan] + [TR[i] - TR[i-1] for i in range(1, len(TR))]
    
#volumn weights
def obv(data):
    """On Balance Volumn: Relates trading volumn to price change.
    OBV(t)=OBV(t-1)+/-Volume(t)
    https://www.investopedia.com/terms/o/onbalancevolume.asp
    """
    priceDir = (data['Close'] > data['Close'].shift(1)).tolist()
    priceEql = (data['Close'] == data['Close'].shift(1)).tolist()
    OBV = data['Volume'].tolist()
    for i in range(1, len(OBV)):
        if priceEql[i]:
            OBV[i] = OBV[i-1]
        elif priceDir[i]:
            OBV[i] += OBV[i-1]
        else:
            OBV[i] = OBV[i-1] - OBV[i]
    return OBV
    
    
#volatility signal  
def atr(data, n):
    """Average True Range: Shows volatility of market.
    ATR(t)=((n-1)*ATR(t-1)+Tr(t))/n 
    where Tr(t)=Max(Abs(High-Low), Abs(High-Close(t-1)), Abs(Low-Close(t-1))
    """
    Tr = [max(x, y, z) for x,y,z in zip(abs(data['High'] - data['Low']), 
                                     abs(data['High'] - data['Close'].shift(1)), 
                                     abs(data['Low'] - data['Close'].shift(1)))]
    ATR = [((n-1)*0 + Tr[0])/n] * len(Tr)
    for i in range(1, len(ATR)):
        ATR[i] = ((n-1)*ATR[i-1] + Tr[i])/n
    return ATR
    
'BBANDSUPPER', 'BBANDSMIDDLE',
'BBANDSLOWER'