import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


#____________IMPORTS____________#
def get_df(TICKER):
    '''
    input : a ticker
    output : pandas dataframe with :
        - open, high, low, close, adjusted close, volume as colums
        - dates as index
    NB : the csv extract from which we read the data go from 1/1/2012 to 12/31/2017
    '''
    path = "/Users/edouardcuny/Desktop/quant/Carmela/data2/" + TICKER
    df = pd.read_csv(path, index_col='Date',dtype={'Adj Close': np.float64}, na_values='null')
    return df

def print_info_stock(dataframe):
    '''
    prints info on the series of the adjusted close of a stock
    visual way to check if everything seems fine
    '''
    ac = dataframe['Adj Close']
    print 'nom      : ' + ac.name
    print 'min date : ' + str(min(ac.index))
    print 'max date : ' + str(max(ac.index))
    print 'nb dates : ' + str(len(ac))
    print 'null     : ' + str(sum(ac.isnull()))

def build_df(TICKER,window_size=10):
    '''
    input : a TICKER
    output : an 'ML ready' dataframe w/ columns in the categories :
        - volume
        - bollinger
        - momentum
        - exotic averages
    '''


    df = get_df(TICKER)
    df.dropna(inplace=True)

    features = []
    features.append(df['Adj Close'])
    features.append(df_bollinger_features(df['Adj Close'],window_size))
    features.append(df_momentum(df['Adj Close']))
    features.append(df_volume(df['Volume'],window_size))
    features.append(df_moving_average(df['Adj Close'],window_size))
    features.append(y(df['Adj Close']))

    df = pd.concat(features, axis=1)
    return df


#____________VOLUME____________#
def df_volume(VOL,window_size=10):
    '''
    input = series of a stock's volum
    output = dataframe w/ columns:
        - vol_mom_1
        - vol_mom_5
        - vol_mom_10
        - vol_rolling_mean
        - vol_spike
        - vol_pr_spike
        - vol_spike_derivative
        the volume momentums over the last 1, 5 and 10 days
    MANUALLY CHECKED : OK
    '''

    # momentum
    vol_mom_1 = (VOL/VOL.shift(1)-1)*100
    vol_mom_5 = (VOL/VOL.shift(5)-1)*100
    vol_mom_10 = (VOL/VOL.shift(10)-1)*100

    # spikes
    vol_rolling_mean = VOL.rolling(window=window_size).mean()
    vol_spike = VOL/vol_rolling_mean
    vol_pr_spike = vol_spike.shift(1)
    vol_spike_derivative = vol_spike - vol_pr_spike

    # rename columns
    vol_mom_1 = vol_mom_1.rename('vol_mom_1')
    vol_mom_5 = vol_mom_5.rename('vol_mom_5')
    vol_mom_10 = vol_mom_10.rename('vol_mom_10')
    vol_rolling_mean = vol_rolling_mean.rename('vol_rolling_mean')
    vol_spike = vol_spike.rename('vol_spike')
    vol_pr_spike = vol_pr_spike.rename('vol_pr_spike')
    vol_spike_derivative = vol_spike_derivative.rename('vol_spike_derivative')

    # list of all vol features
    vol_features = []
    vol_features.append(vol_mom_1)
    vol_features.append(vol_mom_5)
    vol_features.append(vol_mom_10)
    vol_features.append(vol_rolling_mean)
    vol_features.append(vol_spike)
    vol_features.append(vol_pr_spike)
    vol_features.append(vol_spike_derivative)

    return pd.concat(vol_features,axis=1)

#____________EXOTIC MOVING AVERAGE____________#
def df_moving_average(stock, window_size=10):
    '''
    input = series of a stock's adjusted close
    output = dataframe w/ columns:
        - weighted_average
        - wa_spike
        - wa_pr_spike
        - wa_spike_derivative
        - exp_average
        - exp_spike
        - exp_pr_spike
        - exp_spike_derivative
    MANUALLY CHECKED : OK
    '''

    # weighted average
    f = lambda x: np.average(x,weights=range(window_size))
    weighted_average = stock.rolling(window_size).apply(f)
    wa_spike = stock/weighted_average
    wa_pr_spike = wa_spike.shift(1)
    wa_spike_derivative = wa_spike - wa_pr_spike

    # exponential average
    k = 2./(window_size+1)
    l = []
    l.append(stock[0])
    for i in range(1,len(stock)):
        ema = round(stock[i]*k+l[-1]*(1-k),2)
        l.append(ema)
    exp_average = pd.Series(l,index=stock.index)
    exp_spike = stock/exp_average
    exp_pr_spike = exp_spike.shift(1)
    exp_spike_derivative = exp_spike - exp_pr_spike

    # renaming columns
    weighted_average = weighted_average.rename('weighted_average')
    wa_spike = wa_spike.rename('wa_spike')
    wa_pr_spike = wa_pr_spike.rename('wa_pr_spike')
    wa_spike_derivative = wa_spike_derivative.rename('wa_spike_derivative')
    exp_average = exp_average.rename('exp_average')
    exp_spike = exp_spike.rename('exp_spike')
    exp_pr_spike = exp_pr_spike.rename('exp_pr_spike')
    exp_spike_derivative = exp_spike_derivative.rename('exp_spike_derivative')

    # list of all vol features
    features = []
    features.append(weighted_average)
    features.append(wa_spike)
    features.append(wa_pr_spike)
    features.append(wa_spike_derivative)
    features.append(exp_average)
    features.append(exp_spike)
    features.append(exp_pr_spike)
    features.append(exp_spike_derivative)

    return pd.concat(features,axis=1)


#____________BOLLINGER____________#
def df_bollinger_features(stock, window_size):
    '''
    input = series of a stock's adjusted close
    output = dataframe w/ columns:
        - in_BB
        - pr_in_BB
        - out_to_in_BB
        - rolling_mean
        - spike
        - pr_spike
        - spike_derivative
        - crossed_RM_up
        - crossed_RM_down
    MANUALLY CHECKED : OK
    '''

    # BOLLINGER BANDS
    rolling_mean = stock.rolling(window=window_size).mean()
    rolling_std = stock.rolling(window=window_size).std()
    upper_bb = rolling_mean + 2*rolling_std
    lower_bb = rolling_mean - 2*rolling_std

    # plot of BOLLINGER BANDS
    '''
    ax = stock[:100].plot()
    rolling_mean[:100].plot(ax=ax)
    upper_bb[:100].plot(ax=ax, color='c')
    lower_bb[:100].plot(ax=ax, color='c')
    plt.show()
    '''

    # inside BB
    in_BB = (stock < upper_bb) & (stock > lower_bb)
    in_BB[:window_size] = np.NaN

    # previous inside BB
    pr_in_BB = in_BB.shift(1)

    # outside to inside BB
    out_to_in_BB = (pr_in_BB == 0) & (in_BB == 1)
    out_to_in_BB[:window_size+1] = np.NaN


    # Adjusted Close / RM
    spike = stock/rolling_mean
    pr_spike = spike.shift(1)
    spike_derivative = spike - pr_spike
    crossed_RM_up = (pr_spike < 1) & (spike > 1)
    crossed_RM_down = (pr_spike > 1) & (spike < 1)
    crossed_RM_up[:window_size] = np.NaN
    crossed_RM_down[:window_size] = np.NaN

    # renaming columns
    stock = stock.rename('Adj_Close')
    in_BB = in_BB.rename('in_BB')
    pr_in_BB = pr_in_BB.rename('pr_in_BB')
    rolling_mean = rolling_mean.rename('rolling_mean')
    out_to_in_BB = out_to_in_BB.rename('out_to_in_BB')
    spike = spike.rename('spike')
    pr_spike = pr_spike.rename('pr_spike')
    spike_derivative = spike_derivative.rename('spike_derivative')
    crossed_RM_up = crossed_RM_up.rename('crossed_RM_up')
    crossed_RM_down = crossed_RM_down.rename('crossed_RM_down')


    stock_df = pd.concat([in_BB,pr_in_BB,out_to_in_BB,rolling_mean,spike,pr_spike,spike_derivative,crossed_RM_up,crossed_RM_down], axis=1)
    return stock_df

#____________MOMENTUM____________#
def df_momentum(stock):
    '''
    input = series of a stock's adjusted close
    output = dataframe w/ columns:
        - mom_1
        - mom_5
        - mom_10
    MANUALLY CHECKED : OK
    '''

    # compute momentums
    mom_1 = (stock/stock.shift(1)-1)*100
    mom_5 = (stock/stock.shift(5)-1)*100
    mom_10 = (stock/stock.shift(10)-1)*100

    # rename columns
    mom_1 = mom_1.rename('mom_1')
    mom_5 = mom_5.rename('mom_5')
    mom_10 = mom_10.rename('mom_10')

    return pd.concat([mom_1,mom_5,mom_10],axis=1)

#____________Y____________#
def y(stock):
    '''
    input = series of a stock's adjusted close
    output = series of the cumulative return 5 days from now
    MANUALLY CHECKED : OK
    '''

    y = (stock.shift(-5)/stock-1)*100
    y = y.rename('y')
    return y


#____________ML_DATA_PREP____________#
def split_train_test(df1):
    '''
    input : pandas dataframe of a time series of technical features for a stock
    output : X_train,X_test,Y_train,Y_test
    NB : a lot of things to change depending on the features you chose
    '''
    df = df1.copy()

    # droping useless columns
    df.drop([df.columns[0],'rolling_mean'],axis=1,inplace=True)

    # removing rows w/ NaN i.e. first rows and last rows based on window_size
    df.dropna(inplace=True)

    # splitting X and Y
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]

    # features rescaling
    from sklearn.preprocessing import scale
    X.loc[:,'spike'] = scale(X['spike'])
    X.loc[:,'pr_spike'] = scale(X['pr_spike'])
    X.loc[:,'spike_derivative'] = scale(X['spike_derivative'])
    X.loc[:,'mom_1'] = scale(X['mom_1'])
    X.loc[:,'mom_5'] = scale(X['mom_5'])
    X.loc[:,'mom_10'] = scale(X['mom_10'])

    # train & test
    split = 0.7
    n = int(0.7*df.shape[0])
    X_train = X.iloc[:n,:]
    X_test = X.iloc[n:,:]
    Y_train = Y[:n]
    Y_test = Y[n:]

    return X_train, X_test, Y_train, Y_test


#____________RESULTS____________#
def mae(ypred,ytrue):
    return np.abs(ypred - ytrue).mean()

def mse(ypred,ytrue):
    return ((ypred - ytrue)**2).mean()

def print_results_report(Y_pred,Y_test):
    '''
    input : a prediction and ground truth labels
    output :
        - a plot of the first 100 predictions vs truth
        - a % of the time we got the direction right (up or down)
        - Mean Average Error
        - Mean Squared Error
    '''

    comparison = pd.concat([Y_pred,Y_test],axis=1)
    comparison.columns = ['pred','real']
    comparison[:100].plot()
    plt.show()

    print ''

    direction = sum(comparison.iloc[:,0]*comparison.iloc[:,1]>0)/float(comparison.shape[0])
    print str(int(direction*100)) + '% of the time we get the direction right'
    print ''


    print 'MAE TEST  : ' + str(mae(Y_pred,Y_test))
    print 'MSE TEST  : ' + str(mse(Y_pred,Y_test))


#____________LEARNING CURVE____________#
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")


    return plt

def compute_learning_curve(clf,X,Y):
    title = "Learning Curves"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plot_learning_curve(clf, title, X, Y, cv=cv, n_jobs=4)
    plt.show()
