import pandas as pd
import numpy as np
import math

#____________UTILS____________#
def ema(stock,window_size):
    '''
    input =
        - series of a stock's adjusted close
        - a window_size
    output =
        - a series of the exponential moving average

    IMPORTANT
    we ignore nan
    '''
    k = 2./(window_size+1)
    l_return = []
    l_values = []
    for i in range(0,len(stock)-1):
        if len(l_values)==0 and not math.isnan(stock[i]):
            l_return.append(stock[i])
            l_values.append(stock[i])
        if math.isnan(stock[i]):
            l_return.append(float('nan'))
        else:
            if len(l_values)==0:
                l_return.append(stock[i])
                l_values.append(stock[i])
            else:
                ema = stock[i]*k+l_values[-1]*(1-k)
                l_return.append(ema)
                l_values.append(ema)
    return pd.Series(l_return,index=stock.index)

#____________VOLUME____________#
def df_volume(VOL,window_size=12):
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
def df_moving_average(stock, window_size):
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
        + suffix 'window_size' at the end of those names
    MANUALLY CHECKED : OK
    '''

    # weighted average
    f = lambda x: np.average(x,weights=range(window_size))
    weighted_average = stock.rolling(window_size).apply(f)
    wa_spike = stock/weighted_average
    wa_pr_spike = wa_spike.shift(1)
    wa_spike_derivative = wa_spike - wa_pr_spike

    # exponential average
    exp_average = ema(stock,window_size)
    exp_spike = stock/exp_average
    exp_pr_spike = exp_spike.shift(1)
    exp_spike_derivative = exp_spike - exp_pr_spike

    # renaming columns
    weighted_average = weighted_average.rename('weighted_average_'+str(window_size))
    wa_spike = wa_spike.rename('wa_spike_'+str(window_size))
    wa_pr_spike = wa_pr_spike.rename('wa_pr_spike_'+str(window_size))
    wa_spike_derivative = wa_spike_derivative.rename('wa_spike_derivative_'+str(window_size))
    exp_average = exp_average.rename('exp_average_'+str(window_size))
    exp_spike = exp_spike.rename('exp_spike_'+str(window_size))
    exp_pr_spike = exp_pr_spike.rename('exp_pr_spike_'+str(window_size))
    exp_spike_derivative = exp_spike_derivative.rename('exp_spike_derivative_'+str(window_size))

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

#____________MACAD____________#
def df_macad(stock):
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
        (for a window of 12 and 26, window_size as a suffix)

        - ema26_minus_ema12
        - ema9_of_ema_diff
        - macad
        - pr_macad
        - macad_derivative
    '''
    df = pd.concat([df_moving_average(stock,12),df_moving_average(stock,26)], axis=1)
    df['ema26_minus_ema12'] = df['exp_average_26']-df['exp_average_12']
    df['ema9_of_ema_diff'] = ema(df['ema26_minus_ema12'],9)
    df['macad'] = df['ema26_minus_ema12'] - df['ema9_of_ema_diff']
    df['pr_macad'] = df['macad'].shift(1)
    df['macad_derivative'] = df['macad'] - df['pr_macad']
    return df

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
        - rsi
    MANUALLY CHECKED : OK
    '''

    # compute momentums
    mom_1 = (stock/stock.shift(1)-1)*100
    mom_5 = (stock/stock.shift(5)-1)*100
    mom_10 = (stock/stock.shift(10)-1)*100

    # compute rsi
    fup = lambda x : x[x>0].mean()
    fdown = lambda x : np.abs(x[x<0].mean())
    up = mom_1.rolling(window=14).apply(fup)
    down = mom_1.rolling(window=14).apply(fdown)
    rs = up/down
    rsi = 100-100/(1+rs)

    # rename columns
    mom_1 = mom_1.rename('mom_1')
    mom_5 = mom_5.rename('mom_5')
    mom_10 = mom_10.rename('mom_10')
    rsi = rsi.rename('rsi')

    return pd.concat([mom_1,mom_5,mom_10,rsi],axis=1)

#____________STOCK KD____________#

def df_KD_Larry(df):
    '''
    input = df of a stock w/ High, Low, Close
    output = dataframe w/ columns:
        - Kfast
        - Dfast
        - Dslow
        - LWR
    '''
    # compute
    high = df['High'].rolling(window=14).max()
    low = df['Low'].rolling(window=14).min()
    Kfast = 100*(df['Close']-low)/(high-low)
    Dfast = Kfast.rolling(window=3).mean()
    Dslow = Dfast.rolling(window=3).mean()
    LWR = 100*(high-df['Close'])/(high-low)

    #rename
    Kfast = Kfast.rename('Kfast')
    Dfast = Dfast.rename('Dfast')
    Dslow = Dslow.rename('Dslow')
    LWR = LWR.rename('LWR')

    return pd.concat([Kfast,Dfast,Dslow,LWR],axis=1)

#____________ CMF ____________#
def df_CMF(df):
    '''
    input = df of a stock w/ High, Low, Close
    output = dataframe w/ columns:
        - Chainkin Money Flow
    '''
    # compute
    MF_multiplier = (2*df['Close']-df['Low']-df['High'])/(df['High']-df['Low'])
    MF_volume = MF_multiplier*df['Volume']
    CMF = MF_volume.rolling(window=20).sum()/df['Volume'].rolling(window=20).sum()

    # rename
    CMF = CMF.rename('CMF')
    return CMF

#____________ SMI ____________#
def df_SMI(df):
    '''
    input = df of a stock w/ High, Low, Close
    output = dataframe w/ column:
        - SMI : stochastic momentum index
    '''
    # compute
    high = df['High'].rolling(window=14).max()
    low = df['Low'].rolling(window=14).min()
    C = (high+low)/2
    H = df['Close'] - C
    HS1 = ema(H,14)
    HS2 = ema(HS1,14)
    DHL1 = ema(high-low,14)
    DHL2 = ema(DHL1,14)
    SMI = 100*HS2/DHL2

    # rename
    SMI = SMI.rename('SMI')
    return SMI



#____________Y____________#
def y(stock):
    '''
    input = series of a stock's adjusted close
    output = series of the cumulative return 5 days from now
    MANUALLY CHECKED : OK
    '''

    y = (stock.shift(-5)/stock-1)*100
    y[(y>-2)&(y<2)]=0
    y[y>0]=1
    y[y<0]=-1
    y = y.rename('y')
    return y
