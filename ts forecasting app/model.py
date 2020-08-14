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

#def is_stationary(dict):

 #   dataframe=make_df(dict)
  #  test = adfuller(dataframe)
    #print 'adf: ', test[0] 
    #print 'p-value: ', test[1]
    #print'Critical values: ', test[4]
   # if test[0]> test[4]['5%']: 
        #print 'есть единичные корни, ряд не стационарен'
    #    return False
    #else:
        #print 'единичных корней нет, ряд стационарен'
   #     return True



def make_df(dict):
    df = pd.DataFrame(dict['datapoints'])
    df.drop('unit',axis=1,inplace=True)
    df = df.set_index(['timestamp'])
    df.index = pd.to_datetime(df.index, unit='ms')
    df_resample_60T = pd.DataFrame()
    df_resample_60T['average'] = df.average.resample('60T').sum()
    return df_resample_60T


#class Model:
 #   def __init__(self, model):
  #      self.model=model 
         
   # def predict(self,test):
    #    return self.model.predict(test)

    #def train(self, train):
     #   pass


#test = open(r"C:/Users/Svetlana/Documents/flink_metric_data_new/", "r")

#data_folder = 'C:/Users/Svetlana/Documents/flink_metric_data_new/'
#with open(f"{data_folder}10h4id8fd1ii4477diifdd5d3iig9596_15360_129942_read_records_per_second.json") as f:
    #data = json.load(f)
    #df = pd.DataFrame(data['datapoints'])
    #df.drop('unit',axis=1,inplace=True)
    #df = df.set_index(['timestamp'])
    #df.index = pd.to_datetime(df.index, unit='ms')

#df_resample_60T = pd.DataFrame()
#df_resample_60T['average'] = df.average.resample('60T').sum()
#train,test=train_test_split(df_resample_60T.average, shuffle=False)
#model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=24, damped=True)
#hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
#hw_pred = hw_model.predict(start=test.index[0], end=test.index[-1])
#print(hw_pred)
