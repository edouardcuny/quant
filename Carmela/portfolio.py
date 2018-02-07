import matplotlib.pyplot as plt
import pandas as pd

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

def portfolio_evolution(allocation,funds,df):
    df_2 = df.apply(lambda x: x/x[0]) # normalisation
    df_2 = df_2*allocation
    df_2 = df_2*funds
    df_2['TOTAL']=df_2.sum(axis=1)
    return df_2
