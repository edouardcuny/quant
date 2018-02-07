import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visual_check(df):
    '''
    we plot the evolution of the Adj Close for all the tickers
    to visually see if there is something coherent
    we also print the number of rows per ticker to see if we can use all of them
    '''
    tickers = list(df['Ticker'].value_counts().index)

    for ticker in tickers:
        df_stock = df.loc[df['Ticker']==ticker,'Adj Close']
        df_stock.plot(title=ticker)
        plt.show()

    print df['Ticker'].value_counts()

def build_portfolio(df):

    # we drop useless columns
    to_drop = ['KER', 'ENG', 'SGO', 'SAN', 'FR', 'LHN']
    df = df[~df['Ticker'].isin(to_drop)]

    # we only keep the columns that we wanted
    to_keep = ['FP','SAN','BNP','MC','SU','AI','CS','MT','GLE','AIR','OR','BN','DG','VIE','VIV']
    df = df[df['Ticker'].isin(to_keep)]
    print

    # we build the DF with consecutive joins
    tickers = list(df['Ticker'].value_counts().index)
    print tickers
    BN = df.loc[df['Ticker']=='BN','Adj Close']
    GLE = df.loc[df['Ticker']=='GLE','Adj Close']
    pf = pd.DataFrame(BN).join(GLE,lsuffix='BN')
    tickers.remove('BN')
    tickers.remove('GLE')

    pr_ticker = 'GLE'
    for ticker in tickers :
        pf = pf.join(df.loc[df['Ticker']==ticker,'Adj Close'],lsuffix=pr_ticker)
        pr_ticker = ticker

    pf = pf.fillna(method='ffill')
    return pf

def build_pred(df):

    # we drop useless columns
    to_drop = ['KER', 'ENG', 'SGO', 'SAN', 'FR', 'LHN']
    df = df[~df['Ticker'].isin(to_drop)]

    # we only keep the columns that we wanted
    to_keep = ['FP','SAN','BNP','MC','SU','AI','CS','MT','GLE','AIR','OR','BN','DG','VIE','VIV']
    df = df[df['Ticker'].isin(to_keep)]
    print

    # we build the DF with consecutive joins
    tickers = list(df['Ticker'].value_counts().index)
    print tickers
    BN = df.loc[df['Ticker']=='BN','pred']
    GLE = df.loc[df['Ticker']=='GLE','pred']
    pf = pd.DataFrame(BN).join(GLE,lsuffix='BN')
    tickers.remove('BN')
    tickers.remove('GLE')

    pr_ticker = 'GLE'
    for ticker in tickers :
        pf = pf.join(df.loc[df['Ticker']==ticker,'pred'],lsuffix=pr_ticker)
        pr_ticker = ticker

    pf = pf.fillna(method='ffill')
    return pf

def build_alloc(pred):
    alloc = 1./pred.shape[1]*np.ones([pred.shape[0],pred.shape[1]])
    i = 1
    while i < alloc.shape[0]-1:
        indices_0 = [j for j, x in enumerate(pred.iloc[i,:]) if x == 0]
        indices_1 = [j for j, x in enumerate(pred.iloc[i,:]) if x == 1.0]
        indices_m_1 = [j for j, x in enumerate(pred.iloc[i,:]) if x == -1.0]

        if 1.0 in list(pred.iloc[i,:]):
            alloc[i,indices_1] = 1./len(indices_1)
            alloc[i,indices_0] = 0

        else :
            alloc[i,indices_0] = 1./len(indices_0)

        alloc[i,indices_m_1] = 0
        j=1
        while i+j<alloc.shape[0] or j<5 :
            alloc[i+j,:] = alloc[i,:]
            j+=1
        i+=5
    return alloc

def portfolio_evolution(allocation,funds,df):
    df_2 = df.apply(lambda x: x/x[0]) # normalisation
    df_2 = df_2*allocation
    df_2 = df_2*funds
    df_2['TOTAL']=df_2.sum(axis=1)
    return df_2

def portfolio_evo_moving_alloc(allocation,funds,df):
    '''
    alloc is a matrix of size (nb of trading days, number of stocks)
    it's rows must sum to 1
    '''
    df2 = df / df.shift(1)
    df2.iloc[0,:] = 1
    df3 = df2.copy()
    for i in range(df2.shape[0]):
        if i==0:
            cash = funds
        else:
            cash = df3.iloc[i-1,:].sum()
        df3.iloc[i,:] = (cash*allocation[i,:])*df2.iloc[i,:]
    df3['TOTAL'] = df3.sum(axis=1)
    return df3
