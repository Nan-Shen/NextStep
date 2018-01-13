#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:47:51 2018

@author: Nan
"""
from __future__ import division

import numpy as np
import pandas as pd


allstats = ['adjClose', 'Volume', 'OBV', 'Weekday', 'RSI.6', 'RSI.12', 'SMA.3', 
                'EMA.6', 'EMA.12', 'ATR.14', 'MFI.14','ADX.14', 'ADX.20', 'MOM.1', 
                'MOM.3', 'CCI.12', 'CCI.20', 'ROCR.3', 'ROCR.12','MACD', 
                'WILLR', 'TSF.10', 'TSF.20', 'TEMA.3', 'TEMA.6', 'BBANDS']




########################################################
##    features of stockToPredict, NASDAQ, S&P 500  ##
########################################################
def all_features(stocks, 
                 stock_name=['SP&500', 'NASDAQ', 'tg'], 
                 features='all'):
    """stocks:a list of dataframes contain information of different stocks,
    usually including SP&500, NASDQ and target stock
    features: names of features to be included in each stock
    """
    dt = pd.DataFrame()
    
    for s, name in zip(stocks, stock_name):
        dt0 = stock_feature(s, features)
        dt0.columns.values = map(lambda x:('_').join([name, x]), dt0.columns.values)
        dt = pd.concat([dt,dt0],axis=1)  
    return dt

def stock_feature(data, features='all'):
    """
    data: input dataframe
    n: study period
    features: a list of functions for features, e.g. rocr, mom
    """
    func_map = {'adjClose':adjclose, 
                'Volume':volume,
                'Weekday':weekday,
                'OBV':obv, 
                'RSI':rsi, 
                'SMA':sma, 
                'EMA':ema,
                'ATR':atr, 
                'MFI':mfi,
                'ADX':adx, 
                'MOM':mom, 
                'CCI':cci,
                'ROCR':rocr, 
                'MACD':macd, 
                'WILLR':willr, 
                'TSF':tsf, 
                'TEMA':tema, 
                'BBANDS':bbands
                }
    
    f_data = pd.DataFrame()
    if features == 'all':
        features = allstats
    
    for f in allstats:
        if f == 'MACD':
             f_data['outMACD'], f_data['outMACDSignal'], f_data['outMACDHist'] = macd(data, col='Close')
        elif f == 'BBANDS':
             f_data['midBand'], f_data['upperBand'], f_data['lowerBand'] = bbands(data, col='Close')
        elif '.' in f:
             name, n = f.split('.')
             if name == 'EMA' or name == 'SMA':
                 f_data[name] = func_map[name](data['Close'], int(n))
             else:
                 f_data[name] = func_map[name](data, int(n))
        else:
             f_data[f] = func_map[f](data)
    return f_data


##############################
##  static stock indicators  ##
##############################
def adjclose(data):
    """Return adjusted closing price on each day.
    """
    return data['Adj Close']

def volume(data):
    """Return adjusted Volume on each day.
    """
    return data['Volume']

def weekday(data):
    """Which weekday is the predicted day.
    Prices on Friday tend to rise. Prices on Monday has the least tendency to 
    rise. 
    Frank Cross. The behavior of stock prices on Fridays and Mondays. 
    Financial Analyst Journal Vol. 29 No. 6, pages 67-69.
    """
    return data['Weekday']

##############################
##  time series indicators  ##
##############################
"""A stock technical indicator is a series of data points that are derived by 
applying a function to the price data at time t and study period n.
"""
#price change
def rocr(data, n, col='Close'):
    """Rate of Change: Compute rate of change relative to previous trading 
    intervals.
    (Price(t)/Price(t-n))*100
    data: input dataframe
    n: study period
    return a list of ROCR (may contain NA)
    """
    return data[col] / data[col].shift(n) * 100
    
def mom(data, n, col='Close'):
    """Momentum: Measures the change in price.
    Price(t) - Price(t-n)
    """
    return data[col] - data[col].shift(n)

#buy & sell signals
def willr(data):
    """Williams %R: Determine where today’s closing price fell within the
    range of past 10-day’s transaction.
    (highest - Close(t))/(highest - lowest)*100
    """
    highest = data['High'].rolling(10).max()
    lowest = data['Low'].rolling(10).min()
    return (highest - data['Close']) / (highest - lowest) * 100
        
def rsi(data, n, col='Close'):
    """Relative Strength Index: Suggests the overbought and oversold market 
    signal.
    Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100
    Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)-Price(t-1)>0};
           PriceDown(t)=1*(Price(t-1)-Price(t)){Price(t)-Price(t-1)<0};
    n: study period
    """
    deltaPrice = data[col] - data[col].shift(1)
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
    return [np.nan] * (n-1) + [np.mean(price[i-n+1: i+1]) for i in range(n-1, len(price))]

def ema(price, n, Multiplier=False):
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
           ema_init[i] = (price[i] - ema_init[i-1]) * Multiplier + ema_init[i-1]
    return ema_init
      
def macd(data, col='Close'):
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
    price = data[col]
    outMACD = [i - j for i,j in zip(ema(price, 12), ema(price, 26))]
    outMACDSignal = ema(price, 9)
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
    
    plusDI = [100*i/j for i,j in zip(ema(plusDM, n, Multiplier=1/n), atr(data, n))]
    minusDI = [100*i/j for i,j in zip(ema(minusDM, n, Multiplier=1/n), atr(data, n))]
    DX = [100*abs(i-j)/(i+j) for i,j in zip(plusDI, minusDI)]
    return ema(DX, n, Multiplier=1/n)
    
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
    
def tsf(data, n, col='Close'):
    """Time Series Forcasting: Calculates the linear regression of 20-day price.
    Linear Regression Estimate with 20-day price.
    """
    t = range(n)
    t_var = np.var(t, ddof=1)
    rolling_cov = data[col].rolling(n, n).apply(lambda x:np.cov(x, t, ddof=1)[1, 0])
    return rolling_cov / t_var    

#noise elimination and data smoothing    
def tema(data, n):
    """Triple Exponential Moving Average: Smooth the insignificant movements.
    TR(t)/TR(t-1) 
    where TR(t)=EMA(EMA(EMA(Price(t)))) over n days period
    price: a list of prices
    n: study period
    """
    price = data['Close']
    TR = ema(ema(ema(price, n), n), n)
    return [np.nan] + [TR[i] - TR[i-1] for i in range(1, len(TR))]
    
#volumn weights
def obv(data, col='Close'):
    """On Balance Volumn: Relates trading volumn to price change.
    OBV(t)=OBV(t-1)+/-Volume(t)
    https://www.investopedia.com/terms/o/onbalancevolume.asp
    """
    priceDir = (data[col] > data[col].shift(1)).tolist()
    priceEql = (data[col] == data[col].shift(1)).tolist()
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

def bbands(data, col='Close'):
    """Developed by John Bollinger, Bollinger Bands® are volatility bands 
    placed above and below a moving average. Volatility is based on the 
    standard deviation, which changes as volatility increases and decreases. 
    The bands automatically widen when volatility increases and narrow when 
    volatility decreases. 
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_bands
    Middle Band = 20-day simple moving average (SMA)
    Upper Band = 20-day SMA + (20-day standard deviation of price x 2) 
    Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
    """
    midband = sma(data[col], 20)
    upperband = midband + data[col].rolling(20).std() * 2
    lowerband = midband - data[col].rolling(20).std() * 2
    return midband, upperband, lowerband


 