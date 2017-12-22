
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


# In[5]:


ames_train = pd.read_csv('train.csv')
ames_train.columns


# In[15]:


sns.set(style="white", palette="muted", color_codes=True)

#histogram plot
sns.distplot(ames_train['SalePrice'], color='r')


# In[16]:


##rug plot
sns.distplot(ames_train['SalePrice'], hist=False, rug=True, color='m');


# In[19]:


print("Std: %f" % ames_train['SalePrice'].std())
print("Skewness: %f" % ames_train['SalePrice'].skew() )
print( "Kurtosis: %f" % ames_train['SalePrice'].kurt() )


# In[27]:


##build a correlation heatmap
f, ax = plt.subplots(figsize=(12, 9))
correlation_matrix = ames_train.corr()
cols = correlation_matrix.nlargest(12, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(ames_train[cols].values.T)
sns.set(font_scale=1.0)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

