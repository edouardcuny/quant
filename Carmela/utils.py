import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import feature_engineering as fe

#____________IMPORTS____________#
def get_df(TICKER):
    '''
    input : a ticker
    output : pandas dataframe with :
        - open, high, low, close, adjusted close, volume as colums
        - dates as index
    NB : if there is a 0 in the csv we replace it by NA
    we do this bc we had issues w/ volumes = 0 and data that made no sense
    '''
    path = "/Users/edouardcuny/Desktop/quant/Carmela/data2/" + TICKER
    df = pd.read_csv(path, index_col='Date',dtype={'Adj Close': np.float64}, na_values=['null',0])
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

def build_df(TICKER):
    '''
    input : a TICKER
    output : an 'ML ready' dataframe w/ features in features.txt :
    '''

    df = get_df(TICKER)
    rows_before = df.shape[0]
    df.dropna(inplace=True)
    rows_after = df.shape[0]
    print TICKER.split('.')[0] + " - dropped NA : {} >> {}".format(rows_before,rows_after)

    # ADDING FEATURES
    features = []
    features.append(df['Adj Close'])
    features.append(fe.df_bollinger_features(df['Adj Close'],12))
    features.append(fe.df_momentum(df['Adj Close']))
    features.append(fe.df_volume(df['Volume'],12))
    features.append(fe.df_macad(df['Adj Close']))
    features.append(fe.df_KD_Larry(df))
    features.append(fe.df_CMF(df))
    features.append(fe.df_SMI(df))
    features.append(fe.y(df['Adj Close']))

    df = pd.concat(features, axis=1)

    # WE ADD A COLUMN W/ THE TICKER
    df['Ticker'] = TICKER.split('.')[0]

    # Rearrange columns to have ticker first
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    return df

#____________ML_DATA_PREP____________#

def scale(series , mean, std):
    '''
    input : a Series to rescale
    output : the rescaled Series
    '''
    return (series-mean).astype(np.float64)/std

def delete_extreme_value(df):
    '''
    deletes values that are more than 3 std away from the mean
    '''
    rows_before = df.shape[0]
    dftr = df.copy()
    # cb j'en ai ?
    for feature in df.columns:
        if type(feature[0])==str:
            pass
        else:
            std = dftr[feature].std()
            mean = dftr[feature].mean()
            if std == 0:
                pass
            else:
                dftr = dftr.loc[(dftr[feature]<mean+4*std)&(dftr[feature]>mean-4*std),:]
    rows_after = dftr.shape[0]
    print "dropped extreme values : {} >> {}".format(rows_before,rows_after)
    return dftr

def split_train_test(df1,date=None):
    '''
    input :
        - pandas dataframe of a time series of technical features for a stock
        - a date (optional) on which to do the split
    output : X_train,X_test,Y_train,Y_test
    NB : a lot of things to change depending on the features you chose
    '''
    df = df1.copy()

    # removing extreme values
    df = delete_extreme_value(df)

    # removing rows w/ NaN i.e. first rows and last rows based on window_size
    rows_before = df.shape[0]
    df.dropna(inplace=True)
    rows_after = df.shape[0]
    print "dropped NA : {} >> {}".format(rows_before,rows_after)

    # splitting X and Y
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]

    # train & test
    if date is None:
        split = 0.7
        n = int(0.7*df.shape[0])
        X_train = X.iloc[:n,:]
        X_test = X.iloc[n:,:]
        Y_train = Y[:n]
        Y_test = Y[n:]

    else :
        X_train = X.iloc[X.index<date,:]
        X_test = X.iloc[X.index>date,:]
        Y_train = Y[Y.index<date]
        Y_test = Y[Y.index>date]

    # feature rescaling
    bool_features = []
    bool_features.append('in_BB')
    bool_features.append('pr_in_BB')
    bool_features.append('out_to_in_BB')
    bool_features.append('crossed_RM_up')
    bool_features.append('crossed_RM_down')
    bool_features.append('Ticker') # not a bool but not rescalable...
    bool_features.append('Adj Close') # same 

    for column in X.columns:
        if column not in bool_features:
            mean = X_train.loc[:,column].mean()
            std = X_train.loc[:,column].std()
            X_train.loc[:,column] = scale(X_train.loc[:,column], mean, std)
            X_test.loc[:,column] = scale(X_test.loc[:,column], mean, std)

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
