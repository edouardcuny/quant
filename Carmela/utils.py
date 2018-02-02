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

def build_df(TICKER):
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

    return df

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
    bool_features = []
    bool_features.append('in_BB')
    bool_features.append('pr_in_BB')
    bool_features.append('out_to_in_BB')
    bool_features.append('crossed_RM_up')
    bool_features.append('crossed_RM_down')

    from sklearn.preprocessing import scale
    for column in X.columns:
        if column not in bool_features:
            X.loc[:,column] = scale(X[column])
    
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
