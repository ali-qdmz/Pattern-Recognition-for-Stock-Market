# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:12:20 2020

@author: Ali
"""
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import numpy
import numpy as np
import math
import pandas as pd  
import math as m
import gc
from plotly.graph_objs import *
import plotly.graph_objects as go

import plotly as py

from binance_api import *
import matplotlib.pyplot as plt
import datetime
from binance.client import Client

binance_api_key = "V6uqLFl0Up1L9h5Q5ZNMQDdVZzHw2HP7YJwzlpB2NwtyIBdYNPCkNPmpwKPAGMJc"
binance_api_secret = "dR1F06TIb1CYSf9f20CBFjsew1HwAfz8FGeAHvbfkLvWkMWBTaBa07M77id4N1Sz"

binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

#Moving Average  
def MA(df, n):  
    MA = pd.Series(df['Close'].rolling(n).mean(), name = 'MA_' + str(n))  
    df = df.join(MA)  
    return df

#Exponential Moving Average  
def EMA(df, n):  
    EMA = pd.Series(df['Close'].ewm( span = n, min_periods = n - 1).mean(), name = 'EMA_' + str(n))  
    df = df.join(EMA)  
    return df

#Momentum  
def MOM(df, n):  
    M = pd.Series(df['Close'].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df

#Rate of Change  
def ROC(df, n):  
    M = df['Close'].diff(n - 1)  
    N = df['Close'].shift(n - 1)  
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))  
    df = df.join(ROC)  
    return df

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'Close'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n), name = 'ATR_' + str(n))  
    df = df.join(ATR)  
    return df

#Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(df['Close'].rolling(n).mean())  
    MSD = pd.Series(df['Close'].rolling(n).std())  
    b1 = 2 * MSD / MA  
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
    df = df.join(B1)  
    b2 = (df['Close'] - MA + 1 * MSD) / (2 * MSD)  
    B2 = pd.Series(b2, name = 'Bollingerb_' + str(n))  
    df = df.join(B2)  
    return df

def BBANDS2(df, n):
    
    MA = pd.Series(df['Close'].rolling(n).mean())  
    MSD = pd.Series(df['Close'].rolling(n).std())  
    b1 = MA - (MSD*2)
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
    df = df.join(B1)
    M = MA
    M = pd.Series(M, name = 'BollingerM_' + str(n))
    df = df.join(M)
    b2 = MA + (MSD*2) 
    B2 = pd.Series(b2, name = 'Bollingerb_' + str(n))  
    df = df.join(B2)  
    return df


def ichimoku(df,n):
    nine_period_Close = df['Close'].rolling(window= 9).max()
    nine_period_low = df['Low'].rolling(window= 9 ).min()
    df['tenkan_sen'] = (nine_period_Close + nine_period_low) /2
    period26_Close = df['Close'].rolling(window=26).max()
    period26_low = df['Low'].rolling(window=26).min()
    df['kijun_sen'] = (period26_Close + period26_low) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    period52_Close = df['Close'].rolling( window=52).max()
    period52_low = df['Low'].rolling(window=52).min()
    df['senkou_span_b'] = ((period52_Close + period52_low) / 2).shift(26)
    df['chikou_span'] = df['Close'].shift(-22)
    return df
    

#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['Close'] + df['Low'] + df['Close']) / 3)  
    R1 = pd.Series(2 * PP - df['Low'])  
    S1 = pd.Series(2 * PP - df['Close'])  
    R2 = pd.Series(PP + df['Close'] - df['Low'])  
    S2 = pd.Series(PP - df['Close'] + df['Low'])  
    R3 = pd.Series(df['Close'] + 2 * (PP - df['Low']))  
    S3 = pd.Series(df['Low'] - 2 * (df['Close'] - PP))  
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)  
    df = df.join(PSR)  
    return df

#Stochastic oscillator %K  
def STOK(df):  
    SOk = pd.Series((df['Close'] - df['Low']) / (df['Close'] - df['Low']), name = 'SO%k')  
    df = df.join(SOk)  
    return df

# Stochastic Oscillator, EMA smoothing, nS = slowing (1 if no slowing)  
def STO(df,  nK, nD, nS=1):  
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['Close'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))  
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO%d'+str(nD))  
    SOk = SOk.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    SOd = SOd.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    df = df.join(SOk)  
    df = df.join(SOd)  
    return df

def StochRSI(df, period=14, smoothK=3, smoothD=3):
    
    rsi_low = pd.Series(df['RSI_14'].rolling(9).min(), name = 'rsi_low_' + str(9))
    rsiClose = pd.Series(df['RSI_14'].rolling(9).max(), name = 'rsi_Close_' + str(9))
    rsi = pd.Series(df['RSI_14'])
    SOk = (rsi - rsi_low)/(rsiClose - rsi_low)
    SOd =  pd.Series(SOk.rolling(window=3, center=False).mean(), name = 'SO%d'+str(3))  
    SOk = SOk.rolling(window=3, center=False).mean()  
    SOd = SOd.rolling(window=3, center=False).mean()      
    df['sok'] = SOk * 100
    df['sod'] = SOd * 100
    return df  

def Stoch(close,Close,low, smoothk, smoothd, n):
    lowestlow = pd.Series.rolling(low,window=n,center=False).min()
    CloseestClose = pd.Series.rolling(Close, window=n, center=False).max()
    K = pd.Series.rolling(100*((close-lowestlow)/(CloseestClose-lowestlow)), window=smoothk).mean()
    D = pd.Series.rolling(K, window=smoothd).mean()
    return K, D

# Stochastic Oscillator, SMA smoothing, nS = slowing (1 if no slowing)  
def STO(df, nK, nD,  nS=1):  
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['Close'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))  
    SOd = pd.Series(SOk.rolling(window=nD, center=False).mean(), name = 'SO%d'+str(nD))  
    SOk = SOk.rolling(window=nS, center=False).mean()  
    SOd = SOd.rolling(window=nS, center=False).mean()  
    df = df.join(SOk)  
    df = df.join(SOd)  
    return df  
#Trix  
def TRIX(df, n):  
    EX1 = pd.ewma(df['Close'], span = n, min_periods = n - 1)  
    EX2 = pd.ewma(EX1, span = n, min_periods = n - 1)  
    EX3 = pd.ewma(EX2, span = n, min_periods = n - 1)  
    i = 0  
    ROC_l = [0]  
    while i + 1 <= df.index[-1]:  
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]  
        ROC_l.append(ROC)  
        i = i + 1  
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))  
    df = df.join(Trix)  
    return df

#Average Directional Movement Index  
def ADX(df, n, n_ADX):  
    i = 0  
    UpI = []  
    DoI = []  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'Close') - df.get_value(i, 'Close')  
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'Close'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)  
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))  
    df = df.join(ADX)  
    return df

#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(df['Close'].ewm( span = n_fast, min_periods = n_slow - 1).mean())  
    EMAslow = pd.Series(df['Close'].ewm( span = n_slow, min_periods = n_slow - 1).mean())  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(MACD.ewm( span = 9, min_periods = 8).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df

#Mass Index  
def MassI(df):  
    Range = df['Close'] - df['Low']  
    EX1 = Range.ewm(span = 9, min_periods = 8).mean()
    EX2 = EX1.ewm(span = 9, min_periods = 8).mean()
    Mass = EX1 / EX2  
    MassI = pd.Series(Mass.rolling(25).sum(), name = 'Mass Index')  
    df = df.join(MassI)  
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF  
def Vortex(df, n):  
    i = 0  
    TR = [0]  
    while i < df.index[-1]:  
        Range = max(df.get_value(i + 1, 'Close'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR.append(Range)  
        i = i + 1  
    i = 0  
    VM = [0]  
    while i < df.index[-1]:  
        Range = abs(df.get_value(i + 1, 'Close') - df.get_value(i, 'Low')) - abs(df.get_value(i + 1, 'Low') - df.get_value(i, 'Close'))  
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))  
    df = df.join(VI)  
    return df





#KST Oscillator  
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):  
    M = df['Close'].diff(r1 - 1)  
    N = df['Close'].shift(r1 - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(r2 - 1)  
    N = df['Close'].shift(r2 - 1)  
    ROC2 = M / N  
    M = df['Close'].diff(r3 - 1)  
    N = df['Close'].shift(r3 - 1)  
    ROC3 = M / N  
    M = df['Close'].diff(r4 - 1)  
    N = df['Close'].shift(r4 - 1)  
    ROC4 = M / N  
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))  
    df = df.join(KST)  
    return df

#Relative Strength Index  
def RSI(df, n):  
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'Close') - df.get_value(i, 'Close')  
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(UpI.ewm(span = n, min_periods = n - 1).mean())  
    NegDI = pd.Series(DoI.ewm(span = n, min_periods = n - 1).mean())  
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
    df = df.join(RSI)  
    return df



def computeRSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi



#True Strength Index  
def TSI(df, r, s):  
    M = pd.Series(df['Close'].diff(1))  
    aM = abs(M)  
    EMA1 = pd.Series(pd.ewma(M, span = r, min_periods = r - 1))  
    aEMA1 = pd.Series(pd.ewma(aM, span = r, min_periods = r - 1))  
    EMA2 = pd.Series(pd.ewma(EMA1, span = s, min_periods = s - 1))  
    aEMA2 = pd.Series(pd.ewma(aEMA1, span = s, min_periods = s - 1))  
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))  
    df = df.join(TSI)  
    return df

#Accumulation/Distribution  
def ACCDIST(df, n):  
    ad = (2 * df['Close'] - df['Close'] - df['Low']) / (df['Close'] - df['Low']) * df['Volume']  
    M = ad.diff(n - 1)  
    N = ad.shift(n - 1)  
    ROC = M / N  
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))  
    df = df.join(AD)  
    return df

#Chaikin Oscillator  
def Chaikin(df):  
    ad = (2 * df['Close'] - df['Close'] - df['Low']) / (df['Close'] - df['Low']) * df['Volume']  
    Chaikin = pd.Series(ad.ewm(span = 3, min_periods = 2).mean() - ad.ewm(span = 10, min_periods = 9).mean(), name = 'Chaikin')  
    df = df.join(Chaikin)  
    return df

#Money Flow Index and Ratio  
def MFI(df, n):  
    PP = (df['Close'] + df['Low'] + df['Close']) / 3  
    i = 0  
    PosMF = [0]  
    while i < df.index[-1]:  
        if PP[i + 1] > PP[i]:  
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))  
        else:  
            PosMF.append(0)  
        i = i + 1  
    PosMF = pd.Series(PosMF)  
    TotMF = PP * df['Volume']  
    MFR = pd.Series(PosMF / TotMF)  
    MFI = pd.Series(MFR.rolling(n).mean(), name = 'MFI_' + str(n))  
    df = df.join(MFI)  
    return df

#On-balance Volume  
def OBV(df, n):  
    i = 0  
    OBV = [0]  
    while i < df.index[-1]:  
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') > 0:  
            OBV.append(df.get_value(i + 1, 'Volume'))  
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') == 0:  
            OBV.append(0)  
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') < 0:  
            OBV.append(-df.get_value(i + 1, 'Volume'))  
        i = i + 1  
    OBV = pd.Series(OBV)  
    OBV_ma = pd.Series(OBV.rolling( n).mean(), name = 'OBV_' + str(n))  
    df = df.join(OBV_ma)  
    return df

#Force Index  
def FORCE(df, n):  
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))  
    df = df.join(F)  
    return df

#Ease of Movement  
def EOM(df, n):  
    EoM = (df['Close'].diff(1) + df['Low'].diff(1)) * (df['Close'] - df['Low']) / (2 * df['Volume'])  
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))  
    df = df.join(Eom_ma)  
    return df

#Commodity Channel Index  
def CCI(df, n):  
    PP = (df['Close'] + df['Low'] + df['Close']) / 3  
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI_' + str(n))  
    df = df.join(CCI)  
    return df

#Coppock Curve  
def COPP(df, n):  
    M = df['Close'].diff(int(n * 11 / 10) - 1)  
    N = df['Close'].shift(int(n * 11 / 10) - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(int(n * 14 / 10) - 1)  
    N = df['Close'].shift(int(n * 14 / 10) - 1)  
    ROC2 = M / N  
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))  
    df = df.join(Copp)  
    return df

#Keltner Channel  
def KELCH(df, n):  
    KelChM = pd.Series(pd.rolling_mean((df['Close'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))  
    KelChU = pd.Series(pd.rolling_mean((4 * df['Close'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))  
    KelChD = pd.Series(pd.rolling_mean((-2 * df['Close'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))  
    df = df.join(KelChM)  
    df = df.join(KelChU)  
    df = df.join(KelChD)  
    return df

#Ultimate Oscillator  
def ULTOSC(df):  
    i = 0  
    TR_l = [0]  
    BP_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'Close'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        BP_l.append(BP)  
        i = i + 1  
    UltO = pd.Series((4 * pd.Series(BP_l).rolling(7).sum() / pd.Series(TR_l).rolling(7).sum()) + (2 * pd.Series(BP_l).rolling(14).sum() / pd.Series(TR_l).rolling(14).sum()) + (pd.Series(BP_l).rolling(28).sum() / pd.Series(TR_l).rolling(28).sum()), name = 'Ultimate_Osc')  
    df = df.join(UltO)  
    return df

#Donchian Channel  
def DONCH(df, n):  
    i = 0  
    DC_l = []  
    while i < n - 1:  
        DC_l.append(0)  
        i = i + 1  
    i = 0  
    while i + n - 1 < df.index[-1]:  
        DC = max(df['Close'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])  
        DC_l.append(DC)  
        i = i + 1  
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))  
    DonCh = DonCh.shift(n - 1)  
    df = df.join(DonCh)  
    return df

#Standard Deviation  
def STDDEV(df, n):  
    df = df.join(pd.Series(pd.rolling_std(df['Close'], n), name = 'STD_' + str(n)))  
    return df  


    
def percent_change(starting_point, current_point):
    """
    Computes the percentage difference between two points
    :return: The percentage change between starting_point and current_point
    """
    default_change = 0.00001
    try:
        change = ((float(current_point) - starting_point) / abs(starting_point)) * 100.00
        if change == 0.0:
            return default_change
        else:
            return change
    except:
        return default_change
    
    

class asset:
    
    def __init__(self , path ):
        
        df1 = pd.read_csv(path)
        try:
            df1 = df1[['open', 'high', 'low', 'close' , 'volume']]
        except:
            df1 = df1[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
        df1.columns = ['Open','High','Low', 'Close','Volume']
    
        df1 = df1.dropna()
##        df1['RSI_14'] = computeRSI(df1['Close'], 14)
##        #df1 = MFI(df1,14)
##        #df1 = RSI(df1,14)
        df1 = MACD(df1,12,26)
##        df1 = BBANDS2(df1,21)
##        #df1 = OBV(df1,8)
##        df1 = StochRSI(df1)
# =============================================================================
#         Bollingerb_21,BollingerM_21,BollingerB_21 = talib.BBANDS(df1.Close,21)
#         df1['Bollingerb_21'] = Bollingerb_21
#         df1['BollingerM_21'] = BollingerM_21
#         df1['BollingerB_21'] = BollingerB_21
# =============================================================================
        

        df1 = df1.dropna()
        self.asset_name = path[:-12]
        self.df = df1
##        self.ma = self.df.BollingerM_21.values
##        self.upper_band = self.df.Bollingerb_21.values
##        self.lower_band = self.df.BollingerB_21.values
##        self.rsi = self.df.RSI_14.values
        self.macddiff_12_26 = self.df.MACDdiff_12_26.values
##        self.price = self.df.Close.values
        self.macdsign_12_26 = self.df.MACDsign_12_26.values
        self.macd_12_26 = self.df.MACD_12_26.values
##        self.volume = self.df.Volume.values
##        self.sok = self.df.sok.values
##        self.sod = self.df.sod.values
        self.price_now = 0
        self.buy_price = 0
        self.sell_price = 0
        self.buy_flag = True
        self.sell_flag = False
        self.trade_data = []
        self.buys = []
        self.sells = []
        self.bid_profit = []
        self.log = []
        self.buy_trade_time = datetime.datetime.now().replace(year = 2030)
    def update_data(self):

        gc.collect()
        
        data = get_all_binance(self.asset_name,"1m",save = True)
        
        path = self.asset_name + '-1m-data.csv'
            
        df1 = pd.read_csv(path)
        try:
            df1 = df1[['open', 'high', 'low', 'close' , 'volume']]
        except:
            df1 = df1[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
        df1.columns = ['Open','High','Low', 'Close','Volume']
    
        df1 = df1.dropna()
##        df1['RSI_14'] = computeRSI(df1['Close'], 14)
##        #df1 = MFI(df1,14)
##        #df1 = RSI(df1,14)
        df1 = MACD(df1,12,26)
##        df1 = BBANDS2(df1,21)
##        #df1 = OBV(df1,8)
##        df1 = StochRSI(df1)
# =============================================================================
#         Bollingerb_21,BollingerM_21,BollingerB_21 = talib.BBANDS(df1.Close,21)
#         df1['Bollingerb_21'] = Bollingerb_21
#         df1['BollingerM_21'] = BollingerM_21
#         df1['BollingerB_21'] = BollingerB_21
# =============================================================================
        

        df1 = df1.dropna()
        self.asset_name = path[:-12]
        self.df = df1
##        self.ma = self.df.BollingerM_21.values
##        self.upper_band = self.df.Bollingerb_21.values
##        self.lower_band = self.df.BollingerB_21.values
##        self.rsi = self.df.RSI_14.values
        self.macddiff_12_26 = self.df.MACDdiff_12_26.values
##        self.price = self.df.Close.values
        self.macdsign_12_26 = self.df.MACDsign_12_26.values
        self.macd_12_26 = self.df.MACD_12_26.values
##        self.volume = self.df.Volume.values
##        self.sok = self.df.sok.values
##        self.sod = self.df.sod.values

    def online_backtest(self):
        
        benefit = []
        
        self.bid_profit = []
        
        i = 0
        
        try:
        
            if self.trade_data[-1][1] == 'buy':
                
                self.trade_data.pop()
        except:
            
            pass
        
        for index,value in enumerate(self.trade_data):
            
            if i > 0 and i%2 != 0:
            
                temp = (value[0] / self.trade_data[index - 1][0]) - 1
                
                self.bid_profit.append(temp)
                
                benefit.append(temp)
                
            i = i + 1
            
        benefit = np.array(benefit).sum() * 100
        
        return benefit            
  
          


    def trend_angle(self , i , period=3):
                    
        m1 = (self.ma[i] - self.ma[i-period]) / period
        
        m2 = 0 
        
        theta = abs((m1 - m2) / (1 + m1 * m2))
        
        theta = math.atan(theta) * 180 / math.pi    
        
        if m1 > 0 :
            
            theta = theta
            
        else:
            
            theta = -theta
            
        return theta
    
    
    def band_angle(self , i , period=3):
                    
        m1 = (self.upper_band[i] - self.upper_band[i-period]) / period
        
        m2 = (self.lower_band[i] - self.lower_band[i-period]) / period
        
        theta = abs((m1 - m2) / (1 + m1 * m2))
        
        theta = math.atan(theta) * 180 / math.pi    
        
        if m1 > m2 :
            
            theta = theta
            
        else:
            
            theta = -theta
            
        return theta
    
    
    def stochrsi_status(self , i , period=1):
        
        m2 = (self.sok[i] - self.sok[i-period]) / period
        
        m1 = (self.sod[i] - self.sod[i-period]) / period
        
        theta = abs((m1 - m2) / (1 + m1 * m2))
        
        theta = math.atan(theta) * 180 / math.pi    
        
        if self.sod[i-1] > self.sok[i-1] and m1 < m2 :
            
            return theta , "buy"
        
        elif self.sod[i-1] < self.sok[i-1] and m1 > m2 :
            
            return theta , "sell"
        
        else:
            
            return 0 , "unknown"
        
        
    def macd_status(self , i , period=1):
        
        m1 = (self.macdsign_12_26[i] - self.macdsign_12_26[i-period]) / period
        
        m2 = (self.macd_12_26[i] - self.macd_12_26[i-period]) / period
        
        theta = abs((m1 - m2) / (1 + m1 * m2))
        
        theta = math.atan(theta) * 180 / math.pi    
        
        if self.macdsign_12_26[i] > self.macd_12_26[i] and m1 < m2 :
            
            return theta , "buy"
        
        elif self.macdsign_12_26[i] < self.macd_12_26[i] and m1 > m2 :
            
            return theta , "sell"
        
        else:
            
            return 0 , "unknown"
        
    
    def plot(self , start , end ):

        self.buys_init()
        self.sells_init()
        self.trade_data_init()
        fig = go.Figure()
        df = self.df[start:end]
        
        plot_end = 0
        for index,value in enumerate(self.trade_data):
            if value[0] > start :
                plot_start = index
                break
        for index,value in enumerate(self.trade_data):
            if value[1] > end :
                plot_end = index - 1
                break
        if plot_end == 0:
            plot_end = end
        X = self.trade_data[plot_start:plot_end + 1]
        l = []
        for item in X:
            l.append([item[0]-start,item[1]-start])
        trade_data = l
        print(len(trade_data))
        
        
        df.index = range(len(df))
        magnify = 50
        trace0 = go.Scatter(x = df.index + start,y = df.Close.values, name = 'Price')
        trace1 = go.Scatter(x = df.index + start,y = df.MACD_12_26.values, name = 'MACD_12_26')
        trace2 = go.Scatter(x = df.index + start,y = df.MACDsign_12_26.values, name = 'MACDsign_12_26')
        trace3 = go.Bar(x = df.index + start,y = df.MACDdiff_12_26.values , name = 'MACDdiff_12_26')
        trace4 = go.Scatter(x = df.index + start,y = df.RSI_14.values , name = 'Rsi')
        trace5 = go.Scatter(x = df.index + start,y = df.BollingerB_21.values , name = 'BollingerB_21')
        trace6 = go.Scatter(x = df.index + start,y = df.Bollingerb_21.values  , name = 'Bollingerb_21')
        trace7 = go.Scatter(x = df.index + start,y = df.BollingerM_21.values , name = 'BollingerM_21')
        trace8 = go.Bar(x = df.index + start,y = df.Volume.values , name = 'MACDdiff_12_26')
        trace9 = go.Scatter(x = df.index + start,y = df.sok.values , name = 'sok')
        trace10 = go.Scatter(x = df.index + start,y = df.sod.values , name = 'sod')
        data = [trace0,trace1,trace2,trace3,trace4]
        fig = py.tools.make_subplots(rows=2,cols=1,shared_xaxes=True)
        fig.add_trace(trace0,1,1)
        fig.add_trace(trace9,2,1)
        fig.add_trace(trace10,2,1)
        
# =============================================================================
#         fig.add_trace(trace4,2,1)
#         fig.add_trace(trace1,3,1)
#         fig.add_trace(trace2,3,1)
#         fig.add_trace(trace3,3,1)
        fig.add_trace(trace5,1,1)
        fig.add_trace(trace6,1,1)
        fig.add_trace(trace7,1,1)
#         fig.add_trace(trace8,4,1)
# =============================================================================
        fig.update_xaxes(range=[start, end])
        fig
        if trade_data != []:
                
            for i in range(len(trade_data)):
                fig.add_annotation(
                                go.layout.Annotation(
                                    x=trade_data[i][0] + start,
                                    y=self.price[X[i][0]],
                                    xref="x",
                                    yref="y",
                                    text="buy",
                                    showarrow=True,
                                    font=dict(
                                        family="Courier New, monospace",
                                        size=12,
                                        color="#ffffff"
                                        ),
                                    align="center",
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor="#636363",
                                    ax=0,
                                    ay=30,
                                    bordercolor="#c7c7c7",
                                    borderwidth=2,
                                    borderpad=4,
                                    bgcolor="green",
                                    opacity=0.8
                                    )
                            )
            for i in range(len(trade_data)):    
                fig.add_annotation(
                                go.layout.Annotation(
                                    x=trade_data[i][1] + start,
                                    y=self.price[X[i][1]],
                                    xref="x",
                                    yref="y",
                                    text="sell",
                                    showarrow=True,
                                    font=dict(
                                        family="Courier New, monospace",
                                        size=12,
                                        color="#ffffff"
                                        ),
                                    align="center",
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor="#636363",
                                    ax=0,
                                    ay=-30,
                                    bordercolor="#c7c7c7",
                                    borderwidth=2,
                                    borderpad=4,
                                    bgcolor="red",
                                    opacity=0.8
                                    )
                            )

            temp = X[-1][1]
            for item in self.buys:
                if temp < item:
                    if item < df.index[-1] + start:
                        temp = item
                        break

            if temp != X[-1][1]:
                    
                fig.add_annotation(
                                go.layout.Annotation(
                                    x=temp,
                                    y=self.price[temp],
                                    xref="x",
                                    yref="y",
                                    text="buy",
                                    showarrow=True,
                                    font=dict(
                                        family="Courier New, monospace",
                                        size=12,
                                        color="#ffffff"
                                        ),
                                    align="center",
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor="#636363",
                                    ax=0,
                                    ay=30,
                                    bordercolor="#c7c7c7",
                                    borderwidth=2,
                                    borderpad=4,
                                    bgcolor="green",
                                    opacity=0.8
                                    )
                            )
            
        plot(fig,show_link = True, filename = self.asset_name + '.html')     
        

        
    
    def buys_init(self):
        
        self.buys = []
     
        l = []
        
        for i in range(len(self.df)):

            if (self.price[i] < self.ma[i] or (self.price[i] < self.ma[i] + .1 * self.ma[i] and self.macd_status(i)[0] > 15)) and (self.macdsign_12_26[i] > self.macd_12_26[i]) and (abs(self.macddiff_12_26[i]) < abs(np.array(self.macddiff_12_26[i-6:i])).sum()/2) and (self.macd_status(i)[0] != 0) and self.trend_angle(i) > -3e-07:
            
                l.append(i)
                
        l = list(set(l))
        
        l.sort()
        
        places = []
        
        for index,item in enumerate(l):
            
            if item == l[index-1] + 1:
                
                places.append(item)
                
        for item in places:
            
            l.remove(item)
        
        self.buys = l



    def sells_init(self):

        self.sells = []
    
        l = []
        
        for i in range(len(self.df)):

            if (self.price[i] > self.ma[i]) and (self.macdsign_12_26[i] < self.macd_12_26[i]) and (abs(self.macddiff_12_26[i]) < abs(np.array(self.macddiff_12_26[i-6:i])).sum()/2) and (self.macd_status(i)[0] != 0) and  (not self.trend_angle(i) > 3e-07):
                
                l.append(i)
            
            if self.price[i] > self.ma[i]:
                
                temp = []
                
                for j in range(0,10):
                    
                    if self.upper_band[i-j] - self.upper_band[i-j-3] < 0 and self.upper_band[i-j-3] - self.upper_band[i-j-6] > 0 :
                        
                        temp.append(True)
                        
                if True in temp and self.rsi[i] > 50:
                    
                    l.append(i)
                    
        l = list(set(l))
        
        l.sort()
        
        places = []
        
        for index,item in enumerate(l):
            
            if item == l[index-1] + 1:
                
                places.append(item)
                
        for item in places:
            
            l.remove(item)
            
        self.sells = l

    
    
    def trade_data_init(self):
        

        self.trade_data = []        
        pre_buy = 0
        
        for i in range(len(self.buys)):
            
            for j in range(len(self.sells)):
                
                if self.sells[j] > self.buys[i] and self.buys[i] > pre_buy:
                    
                    self.trade_data.append([self.buys[i],self.sells[j]])
                    
                    pre_buy = self.sells[j]
                    
                    break 
        
        
    def backtest(self , start , end):
        
        self.buys_init()
        self.sells_init()
        self.trade_data_init()
        backtest_end = 0
        for index,value in enumerate(self.trade_data):
            if value[0] > start :
                backtest_start = index
                break
        for index,value in enumerate(self.trade_data):
            if value[1] > end :
                backtest_end = index - 1
                break
        if backtest_end ==0:
            backtest_end = end
        trade_data = self.trade_data[backtest_start:backtest_end+1]

        benefit_percent = []
        
        for item in trade_data:
            
            percent = (self.price[item[1]]/self.price[item[0]] - 1)*100
            
            if percent < -1 : percent = -1
            
            benefit_percent.append(percent)
        
        benefit = np.array(benefit_percent).sum()
        
        print(benefit_percent)
        
        print(trade_data)
        
        print(benefit)
        
        print(benefit/len(trade_data) , "  percent per deal")
        
        print(len(trade_data) , " total trades")
        
        
    def pattern_Recognitor(self):

        gc.collect()
        
        accuracy_array = []
        
        pattern_images_folder = "patterns/"
        
        end_point = 50000 
        
        plot_data = False
        
        samples = 0
        
        pattern_similarity_value = 70
        
        pattern_array = []
        
        performance_array = []
        
        pattern_for_recognition = []
        
        dots_for_pattern = 30
        
        all_data = self.df.Close.values
        
        average_line = all_data[:end_point]
        
        x = len(average_line) - (30 + dots_for_pattern)
    
        y = 1 + dots_for_pattern
        
        for index in reversed(range(1, dots_for_pattern + 1)):
            pattern = percent_change(all_data[- dots_for_pattern - 1],
                                     all_data[- index])
            pattern_for_recognition.append(pattern)
            
        #print(all_data[-1] , all_data[-31])
        
        while y < x:
            
            pattern = []
    
            for index in reversed(range(dots_for_pattern)):
                point = percent_change(average_line[y - dots_for_pattern] , average_line[y - index])
                pattern.append(point)
    
            # Create the pattern array and store it
            pattern_array.append(pattern)
    
            # Take the range of the outcome using 10 values from the 20th after the current point
            outcome_range = average_line[y+4:y+14]
    
            # Take the current point
            current_point = average_line[y]
    
            # Get the average value of the outcome
            try:
                average_outcome = np.average(outcome_range)
            except Exception as e:
                print(e)
                average_outcome = 0
    
            # Get the future outcome for the pattern based on the average outcome value
            future_outcome = percent_change(current_point, average_outcome)
    
            # Store the outcome value
            performance_array.append(future_outcome)
    
            y += 1
        
        found_pattern = False
    
        # Contains the array of patterns to be plotted
        plot_pattern_array = []
    
        # Contains the array of outcomes predicted by the identified patterns
        predicted_outcomes_array = []
    
        for pattern in pattern_array[:-5]:
            # Tells if enough similarities have been found in order to consider the pattern similar to the one currently
            #samples considered
            similarities_are_found = True
    
            # Compute the percent changes for each point of the two patterns, the one that we are considering and the
            # current one obtained from the last 10 entries of the data
            similarities_array = []
            for index in range(dots_for_pattern):
                # Compute the values of similarity only if it's the first value to be computed, or if the previous one was
                # at least 50% similar
                if index == 0 or similarities_array[index - 1] > 0:
                    similarities_array.append(100.00 - abs(percent_change(pattern[index], pattern_for_recognition[index])))
    
                # Otherwise just break the for loop and stop computing similarities
                else:
                    similarities_are_found = False
                    break
    
            # If sufficient similarities were found continue on
            if similarities_are_found:
                # Compute how similar are the two patterns
                how_similar = np.sum(similarities_array) / dots_for_pattern

                if how_similar > pattern_similarity_value:
                    # If a pattern satisfies the similarity value, remember that a pattern was found and append that
                    # pattern to the list of patterns to plot
                    found_pattern = True
                    plot_pattern_array.append(pattern)
    
        prediction_array = []

        gc.collect()
        
        if found_pattern:
            
            # If at least one similar pattern was found then print all the pattern that are in the array of patterns to be
            # plotted
            xp = np.arange(0, dots_for_pattern, 1)
            if plot_data:
                plt.figure(figsize=(10, 6))
    
            for pattern in plot_pattern_array:
                pattern_index = pattern_array.index(pattern)
    
                # Determine the color based on the prediction of the pattern
                if performance_array[pattern_index] > pattern_for_recognition[dots_for_pattern - 1]:
                    # If the prediction of the pattern is greater than the value of the pattern use the green
                    plot_color = '#24BC00'
                    prediction_array.append(1.00)
    
                else:
                    # Otherwise use the red
                    plot_color = '#D40000'
                    prediction_array.append(-1.00)
                predicted_outcomes_array.append(performance_array[pattern_index])
                if plot_data:
                    plt.plot(xp, pattern)
                    predicted_outcomes_array.append(performance_array[pattern_index])
    
                    # Plot the dot representing the value predicted by the pattern
                    # The color of the dot will be red if the outcome is good, and red otherwise
                    plt.scatter(dots_for_pattern + 5, performance_array[pattern_index], c=plot_color, alpha=.3)
    
            # Get the average of 10 future values to determine the chart gait and plot the dot as a reference of what is
            # going to happen
# =============================================================================
##             real_outcome_range = all_data[end_point+20:end_point+30]
##             real_average_outcome = np.average(real_outcome_range)
##             real_movement = percent_change(all_data[end_point], real_average_outcome)
#     
#             if plot_data:
#                 plt.scatter(40, real_movement, s=25, c='#54FFF7')
# =============================================================================
    
            # Get the average of the predicted values and plot a dot representing what the system has predicted will happen
            predicted_average_outcome = np.average(predicted_outcomes_array)
    
            if plot_data:
                plt.scatter(40, predicted_average_outcome, s=25, c='b')
    
            # Also plot the patter that has been recognized with a different line width and color to make it stand out on
            # the graph with all the similar patterns
            if plot_data:
                pass
##                plt.plot(xp, pattern_for_recognition, '#54FFF7', linewidth=3)
##    
##                plt.grid(True)
##                plt.title("Pattern recognition")
##                plt.suptitle("Patterns recognized after {} samples".format(samples))
##                plt.savefig(pattern_images_folder + "patter_recognition_{}_samples.png".format(samples))
    
            #print(prediction_array)
    
            prediction_average = np.average(prediction_array)
            #print(prediction_average)
    
            if prediction_average < 0:
                return "sell" , predicted_average_outcome , prediction_array , pattern_for_recognition
                pass
                #print("Drop predicted")
                #print(pattern_for_recognition[29])
# =============================================================================
#                 print(real_movement)
#                 if real_movement < pattern_for_recognition[29]:
#                     accuracy_array.append(100)
#                 else:
#                     accuracy_array.append(0)
# =============================================================================
    
            if prediction_average > 0:
                #print("Rise predicted ")
                #print(pattern_for_recognition[29])
                return "buy" , predicted_average_outcome , prediction_array , pattern_for_recognition
# =============================================================================
#                 print(real_movement)
#                 if real_movement > pattern_for_recognition[29]:
#                     accuracy_array.append(100)
#                 else:
#                     accuracy_array.append(0)
# =============================================================================

        

        
      
import shelve
import os
import datetime
import requests
from telegram import telegram_bot_sendtext                
                



            
            
            
            
# =============================================================================
# f = open('markets.txt','r')
# s = f.readlines()
# f.close()
# l = []
# for i in range(len(s)):
#     if i%7 ==0:
#         l.append(s[i])
# for i in range(len(l)):
#     l[i] = l[i].replace("\n","")
#     l[i] = l[i].replace("/","")
#     l[i] = l[i].replace(" ","")
# =============================================================================

x = ["BTCUSDT"]

# =============================================================================
# x = ['BTCUSDT']
# =============================================================================

def file_Prep():
    
    global x
    
    #x = ["LINKUSDT"]
        
    preferd_markets = []
    
    for i in range(len(x)):
        
        try:
            
            xf = pd.read_csv(x[i] + '-1m-data.csv')
            
            xf = xf.drop(xf.index[-10:])
            
            xf.to_csv(x[i] + '-1m-data.csv')
            
        except:
            
            data2 = get_all_binance(x[i], "1m" , save = True)
            
            xf = pd.read_csv(x[i] + '-1m-data.csv')
            
            xf = xf.drop(xf.index[-10:])
            
            xf.to_csv(x[i] + '-1m-data.csv')                                


file_Prep()

obs = []

for items in x:
    
    obs.append(asset(items + '-1m-data.csv'))
         
counter = 0
def Main():
    global counter
    
    try:
                        
        global obs
        asset_log = []
        while True:            
            i = 0
            gc.collect()
            
            for item in x:
                obs[i].update_data()
                try:
                    
                    t , predicted_average_outcome , prediction_array , pattern_for_recognition = obs[i].pattern_Recognitor()
                    print(predicted_average_outcome - pattern_for_recognition[-1])
                    telegram_bot_sendtext(obs[0].asset_name + "\n" + "delta is : " + str(predicted_average_outcome - pattern_for_recognition[-1]) + "\n" + "Macd is : " + str(obs[0].df.MACD_12_26.values[-2])  + "\n" + "Macd diff is : " + str(obs[0].df.MACDdiff_12_26.values[-2])  + "\n" + "Macdsign is : " + str(obs[0].df.MACDsign_12_26.values[-2])  + "\n" + str(obs[0].df.Close.values[-5:]),"369692491")
                    telegram_bot_sendtext("=======================","369692491")
                    telegram_bot_sendtext("=======================","369692491")
                    telegram_bot_sendtext("=======================","369692491")
                except:
                    
                    t = ""
         
                if t != "":
                    asset_log.append(True)
                else:
                    asset_log.append("None")
                i += 1
            buy_list = []
# =============================================================================
#         for k in range(1,3):
#             buy_list.append(obs[0].macd_status(-k)[1] )
# =============================================================================
        
            if (obs[0].macd_status(-2)[1] == "buy"):
                if  True in asset_log:
                    print("here we gooooooooooooooooooooooooooo")
                    telegram_bot_sendtext(obs[0].asset_name + "\n" + "delta is : " + str(predicted_average_outcome - pattern_for_recognition[-1]) + "\n" + "Macd is : " + str(obs[0].df.MACD_12_26.values[-2])  + "\n" + "Macd diff is : " + str(obs[0].df.MACDdiff_12_26.values[-2])  + "\n" + "Macdsign is : " + str(obs[0].df.MACDsign_12_26.values[-2])  + "\n" + str(obs[0].df.Close.values[-5:]),"369692491")
                    telegram_bot_sendtext(obs[0].asset_name + "\n" + "delta is : " + str(predicted_average_outcome - pattern_for_recognition[-1]) + "\n" + "Macd is : " + str(obs[0].df.MACD_12_26.values[-2])  + "\n" + "Macd diff is : " + str(obs[0].df.MACDdiff_12_26.values[-2])  + "\n" + "Macdsign is : " + str(obs[0].df.MACDsign_12_26.values[-2])  + "\n" + str(obs[0].df.Close.values[-5:]),"193425555")
                    telegram_bot_sendtext("=======================","369692491")
                    telegram_bot_sendtext("=======================","369692491")
                    telegram_bot_sendtext("=======================","369692491")
                    telegram_bot_sendtext("=======================","193425555")
                    telegram_bot_sendtext("=======================","193425555")
                    telegram_bot_sendtext("=======================","193425555")
                    asset_log.clear()
                    buy_list.clear()

            
# =============================================================================
#                 
#                 obs[i].update_data()
#         
#         
#                 try:
#                     t , predicted_average_outcome , prediction_array , pattern_for_recognition = obs[i].pattern_Recognitor()
#                     print(predicted_average_outcome - pattern_for_recognition[-1])
#                 except:
#                     t = ""
#                 
#                 if obs[i].buy_flag:
#                     
#                     if (t == "buy") or (datetime.datetime.now() > obs[i].buy_trade_time):
#                         if ( predicted_average_outcome - pattern_for_recognition[-1] > 2) or (datetime.datetime.now() > obs[i].buy_trade_time):
#                             if ( predicted_average_outcome - pattern_for_recognition[-1] > 3):
#                                 telegram_bot_sendtext("delta is {}".format(predicted_average_outcome - pattern_for_recognition[-1]),"369692491")
#                                 telegram_bot_sendtext("delta is {}".format(predicted_average_outcome - pattern_for_recognition[-1]),"193425555")
#                             telegram_bot_sendtext("bouth {} on {} at {}\ndelta is {}\nsimilar patterns number {} ".format(obs[i].asset_name,obs[i].df.Close.values[-1],datetime.datetime.now(),predicted_average_outcome - pattern_for_recognition[-1],len(prediction_array)),'193425555')
#                             telegram_bot_sendtext("bouth {} on {} at {}\ndelta is {}\nsimilar patterns number {} ".format(obs[i].asset_name,obs[i].df.Close.values[-1],datetime.datetime.now(),predicted_average_outcome - pattern_for_recognition[-1],len(prediction_array)),'369692491')
#                             telegram_bot_sendtext("last prices are : " + obs[i].asset_name + str(obs[i].df.Close.values[-8:]),"369692491")
#                             print("bouth {} on {} at {}".format(obs[i].asset_name,obs[i].df.Close.values[-1],datetime.datetime.now()))
#                             print("delta is : " , predicted_average_outcome - pattern_for_recognition[-1])
#                             print("len prediction_array is : ",len(prediction_array))
#                             print("=====================================================")
#                             obs[i].log.append([obs[i].df.Close.values[-1],datetime.datetime.now(),"buy"])
#                             sell_time = [datetime.datetime.now().minute+20,datetime.datetime.now().minute+30]
#                             
#                             try:
#                                 os.remove('data.dat')
#                                 os.remove('data.dir')
#                                 os.remove('data.bak')
#                             except:
#                                 print("remove file skipped")
#                             db = shelve.open('data')
#                             db['data'] = obs
#                             db.close()
#                             if sell_time[0] > 59:
#                                 sell_time[0] = sell_time[0]-60
#                             if sell_time[1] > 59:
#                                 sell_time[1] = sell_time[1]-60
#                             t = ""
#                             obs[i].buy_flag = False
#                             obs[i].sell_flag = True
#                         
#                 if obs[i].sell_flag:
#                     if obs[i].buy_trade_time.year == 2030:
#                         profit = .2
#                     else:
#                         profit = .5
#                     
#                     if ((((obs[i].df.High.values[-1] / obs[i].log[-1][0])-1) * 100) > profit) or (((((obs[i].df.Close.values[-1] / obs[i].log[-1][0])-1) * 100) < -1.5)):
#                         obs[i].buy_trade_time = datetime.datetime.now().replace(year = 2030)
#                         if (((((obs[i].df.Close.values[-1] / obs[i].log[-1][0])-1) * 100) < -1.5)):
#                             obs[i].buy_trade_time = datetime.datetime.now() + datetime.timedelta(minutes = 1)
#                             telegram_bot_sendtext("Loss more that 1.5 in {} on {} at {}".format(obs[i].asset_name,obs[i].df.Close.values[-1],datetime.datetime.now()),"369692491")
#                             telegram_bot_sendtext("Loss more that 1.5 in {} on {} at {}".format(obs[i].asset_name,obs[i].df.Close.values[-1],datetime.datetime.now()),"193425555")
#                         telegram_bot_sendtext("sold {} on {} at {}".format(obs[i].asset_name,obs[i].df.Close.values[-1],datetime.datetime.now()),"369692491")
#                         telegram_bot_sendtext("sold {} on {} at {}".format(obs[i].asset_name,obs[i].df.Close.values[-1],datetime.datetime.now()),"193425555")
#                         obs[i].log.append([obs[i].df.Close.values[-1],datetime.datetime.now(),"sell"])
#                         
#                         print("sold {} on {} at {}".format(obs[i].asset_name,obs[i].df.Close.values[-1],datetime.datetime.now()))
#                         print("=====================================================")
#                         try:
#                             os.remove('data.dat')
#                             os.remove('data.dir')
#                             os.remove('data.bak')
#                         except:
#                             print("remove file skipped")
#                         db = shelve.open('data')
#                         db['data'] = obs
#                         db.close()
#                         obs[i].buy_flag = True
#                         obs[i].sell_flag = False                    
#                  
#                         
#                         
#                         
#                     
#                                     
#                 i += 1                                          
# =============================================================================
    
                                                                            
    except:
        print("server restarted")
        telegram_bot_sendtext("Server restarted" , "193425555")
        telegram_bot_sendtext("Server restarted" , "369692491")
##        for i in range(len(obs)):
##            os.remove("C:/Users/Administrator/AppData/Local/Programs/Python/Python37/" + obs[i].asset_name + "-1m-data.csv")

        file_Prep()
        Main()



Main()
