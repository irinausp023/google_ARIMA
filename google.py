#!/usr/bin/env python
# coding: utf-8

# In[18]:


import warnings       
warnings.filterwarnings('ignore')
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# In[19]:


df = pd.read_csv('GOOG.csv', index_col=['Date'], parse_dates=['Date'])


# In[20]:


df.head()


# In[21]:


df.shape


# In[38]:


df.isnull().values.any()


# In[22]:


df['Close'].plot(figsize=(15, 6), linewidth=1, label = 'Closing price');


# In[23]:


result = adfuller(df['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[69]:


diff1 = pd.Series(df['Close'].diff()).dropna()
diff1.plot(figsize=(15, 6), linewidth=1, label = '1st difference')


# In[25]:


result = adfuller(diff1)
print('ADF Statistic: %f' % result[0])
print('p-value: %e' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[48]:


#ACF and PACF of the initial data
plt.rcParams["figure.figsize"] = (15,6)
plot_acf(df.Close);
plot_pacf(df.Close);


# In[49]:


#ACF and PACF of the first difference
plt.rcParams["figure.figsize"] = (15,6)
plot_acf(diff1);
plot_pacf(diff1);


# In[29]:


size = int(len(df) * 0.8)
train_df, test_df = df[0:size], df[size:len(df)]


# In[50]:


def select_mod(p_values, d_values, q_values, train):    
    mod_eval = []
    mod_eval = pd.DataFrame(mod_eval)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    model = ARIMA(train, order=order)
                    model_fit = model.fit()
                    mod_eval = mod_eval.append({'Order':order, 'BIC':model_fit.bic}, ignore_index=True)
                except:
                    continue
    return mod_eval
    


# In[51]:


p = range(0, 2)
d = range(0, 2)
q = range(0, 2)
models = select_mod(p,d,q,train_df['Close'])
models


# In[33]:


print(models[models.BIC == models.BIC.min()])


# In[57]:


model = ARIMA(train_df['Close'], order=(0,1,0))
model_fit = model.fit()
resid = model_fit.resid

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
resid.plot.kde();

plt.subplot(1, 2, 2)
resid.plot()


# In[75]:


k2, p = stats.normaltest(resid)
print("p = {:g}".format(p))


# In[36]:


plt.rcParams["figure.figsize"] = (15,6)
plot_acf(resid);
plot_pacf(resid);


# In[61]:


history = [x for x in train_df['Close']]
predictions = list()
for t in range(len(test_df['Close'])):
    model = ARIMA(history, order=(0,1,0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test_df['Close'][t])


# In[67]:


#Root Mean Square Error
rmse = np.sqrt(mean_squared_error(test_df['Close'], predictions))
#Mean Absolute Percentage Error
mape = np.round(np.mean(np.abs(test_df['Close']-predictions)/test_df['Close'])*100,2)
print('Test RMSE: %.3f' % rmse)
print('Test MAPE: %.3f' % mape)


# In[81]:


plt.figure(figsize=(20,10))
plt.plot(test_df['Close'].index, test_df['Close'], label = 'Actual values')
plt.plot(test_df['Close'].index, predictions, color='red', label = 'Predicted values')
plt.legend(fontsize = 15)


# In[82]:


plt.figure(figsize=(20,10))
plt.plot(train_df['Close'].index, train_df['Close'], label = 'Training set')
plt.plot(test_df['Close'].index, predictions, color='red', label = 'Predicted values')
plt.legend(fontsize = 15)


# In[ ]:




