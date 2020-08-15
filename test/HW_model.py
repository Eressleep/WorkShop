import pandas as pd
import numpy as np
import statsmodels as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score
import json
from sklearn.model_selection import train_test_split


def hw_classifier(dict):

    dataframe=make_df(dict)

    train,test=train_test_split(dataframe.average, shuffle=False)
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=24, damped=True)
    hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
    hw_pred = hw_model.predict(start=test.index[0], end=test.index[-1])
    res=hw_pred.to_dict()
    return (res)


def make_df(dict):
    df = pd.DataFrame(dict['datapoints'])
    df.drop('unit',axis=1,inplace=True)
    df = df.set_index(['timestamp'])
    df.index = pd.to_datetime(df.index, unit='ms')
    df_resample_60T = pd.DataFrame()
    df_resample_60T['average'] = df.average.resample('60T').sum()
    return df_resample_60T


