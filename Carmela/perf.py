import numpy as np
import pandas as pd

def sharpe(stock):
    daily_returns = stock/stock.shift(1)-1
    return round(np.sqrt(252)*daily_returns.mean()/daily_returns.std(ddof=0),3)

def max_drawdown(stock):
    stock = pd.Series(stock)
    max2here = stock.expanding().max()
    return min((stock - max2here)/max2here)

def rolling_max_drawdown(stock,window):
    rolling_stock = stock.rolling(window=window)
    return round(rolling_stock.apply(max_drawdown).min()*100,2)

def annual_return(stock):
    # list the years
    years = sorted(list(set([x.split('-')[0] for x in list(stock.index)])))
    d = {}
    for year in years:
        dates = [x for x in list(stock.index) if x.split('-')[0]==str(year)]
     #if max(dates).split('-')[1]!='12' or max(dates).split('-')[2]!='31' :
     #    print 'WARNING : last date of trading for year {} is {}'.format(year,max(dates))
     #if min(dates).split('-')[1]!='01' or min(dates).split('-')[2]!='02' :
     #    print 'WARNING : first date of trading for year {} is {}'.format(year,min(dates))
        last = stock.loc[stock.index==max(dates)][0]
        first = stock.loc[stock.index==min(dates)][0]
        d[year] = str(int(round((last/first-1)*100,0))) + '%'
    return d

def print_d(dic):
    for key in sorted(dic):
        print key,dic[key]
    print
