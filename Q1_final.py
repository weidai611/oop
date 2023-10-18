#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from datetime import datetime
import datetime as dt
import sys
import numpy as np
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import seaborn as sns
from pylab import rcParams 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from arch import arch_model
from numpy.linalg import LinAlgError
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot, moment
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal
from sklearn.model_selection import TimeSeriesSplit
import warnings


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
sns.set(style="darkgrid", color_codes=True)
rcParams['figure.figsize'] = 8,4


# In[15]:


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)
    
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


# In[16]:


def plot_correlogram(x, lags=None, title=None):    
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values),2)}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)


# In[ ]:





# In[133]:


data = pd.read_csv("C:/Users/freedom1106/HistoricalData.csv")


# In[6]:


data.head(15)


# In[79]:


# Calculate monthly returns as percentage price changes
data['Return'] = 100 * (data['SP500'].pct_change())
data['Log_Return'] = np.log(data['SP500']).diff().mul(100) # rescale to faciliate optimization
data.dropna(inplace=True)

# Plot ACF, PACF and Q-Q plot and get ADF p-value of series
plot_correlogram(data['Log_Return'], lags=30, title='SP500 (Log, Diff)')


# In[59]:


# Plot ACF, PACF and Q-Q plot and get ADF p-value of series
plot_correlogram(data['VIX'], lags=30, title='VIX raw')


# In[60]:


plot_correlogram(data['Log_Return'].sub(data['Log_Return'].mean()).pow(2), lags=30, title='SP500 monthly Volatility')


# In[61]:


#compute the monthly volatility as the standard deviation of price returns. 
#Then convert the monthly volatility to quarterly volatility.
# Calculate dmonthy std of returns
std_monthly = data['Return'].std()
print(f'Monthly volatility: {round(std_monthly,2)}%')

# Convert daily volatility to monthly volatility
std_quarterly = np.sqrt(3) * std_monthly
print(f'\nQuarterly volatility: {round(std_quarterly,2)}%')


# In[51]:


get_ipython().system('pip install arch -U')


# In[62]:


from arch import arch_model


# In[63]:


#define a basic GARCH(1,1) model
# Specify GARCH model assumptions
gh= arch_model(data['Return'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'skewt')
# Fit the model
gh_result = gh.fit(disp = 'off')


# In[64]:


# Get model estimated volatility
skew_volatility = gh_result.conditional_volatility


# In[65]:


# Display model fitting summary
print(gh_result.summary())


# In[66]:


# Plot model fitting results
plt.plot(skew_volatility, color = 'red', label = 'Skewed-t Volatility')
plt.plot(data['Return'], color = 'grey', 
         label = 'monthly Returns', alpha = 0.4)
# Plot EGARCH  estimated volatility
plt.plot(data['VIX'], color = 'blue', label = 'VIX')
plt.legend(loc = 'upper right')
plt.show()


# In[67]:


# Make 5-period ahead forecast
gh_forecast = gh_result.forecast(horizon = 10)

# Print the forecast variance
print(gh_forecast.variance[-1:])


# In[ ]:





# In[77]:


data.head(5)


# In[83]:


VIX=data.VIX
R_SP500=data.drop(['SP500','NASDAQ','LIBOR3M','VIX','CPI','CDS5Y','HPI','Log_Return'],axis=1)


# In[84]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# Create training and test data
#
R_SP500_train,R_SP500_test,VIX_train,VIX_test=train_test_split(R_SP500,VIX,test_size=0.2)
R_SP500_train.head()


# In[69]:


import statsmodels.formula.api as smf


# In[86]:


R_SP500_train['VIX'] = data.VIX


# In[87]:


R_SP500_train.head()


# In[101]:


# create and fit the linear model
lm = smf.ols(formula="VIX ~ Return", data=R_SP500_train)
result=lm.fit()


# In[104]:


result.summary()


# In[105]:


# prediction
result.predict(R_SP500_test)


# In[115]:





# In[120]:


# create and fit the linear model
lm2 = smf.ols(formula="VIX ~ Return", data=data)
result2=lm2.fit()
result2.predict(data)


# In[122]:


# Plot model fitting results
plt.plot(result2.predict(data), color = 'red', label = 'OLS')
plt.plot(data['Return'], color = 'grey', 
         label = 'monthly Returns', alpha = 0.4)
# Plot EGARCH  estimated volatility
plt.plot(data['VIX'], color = 'blue', label = 'VIX')
plt.legend(loc = 'upper right')
plt.show()


# In[138]:


#Quarterly model
#convert monthly data to quarterly data
Data = pd.read_csv("C:/Users/freedom1106/HistoricalData.csv")
# convert integers to datetime format
Data['date'] = pd.to_datetime(Data['date'], format='%Y%m')
print(Data)
Data['date'] = Data['date'].astype('datetime64[ns]')
Data.set_index('date').resample('Q').agg(['mean', 'median', 'std'])


# In[139]:


Data_final=Data.set_index('date').resample('Q').agg(['mean', 'median', 'std'])


# In[140]:


Data_final.head()


# In[141]:


# Calculate monthly returns as percentage price changes
Data['Return'] = 100 * (data['SP500'].pct_change())
Data['Log_Return'] = np.log(data['SP500']).diff().mul(100) # rescale to faciliate optimization


# In[143]:


# create and fit the linear model
lm_f = smf.ols(formula="VIX ~ Return", data=Data)
result_f=lm_f.fit()
ped=result_f.predict(Data)


# In[144]:


ped.head(10)


# In[ ]:




