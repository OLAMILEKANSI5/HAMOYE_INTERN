#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# In[28]:


df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')
# show all columns in pandas
pd.set_option('display.max_columns', None)
df.head()


# In[29]:


df.drop(['lights', 'date'], axis=1, inplace=True)

scaler = MinMaxScaler()

normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_new = normalised_df.copy()


# # Question 12

# In[7]:


# reshape features to 2d array to signify single feature.
features = normalised_df['T2'].values.reshape(-1, 1)
output = normalised_df['T6']


# In[8]:


linear_model = LinearRegression()
linear_model.fit(features, output)


# In[12]:



rsquared = linear_model.score(features, output)
print("R squared score: ", round(rsquared, 2))


# In[42]:


features_df = df_new.drop('Appliances', axis='columns')
heating_target = df_new['Appliances']

X_train, X_test, y_train, y_test = train_test_split(features_df, heating_target, test_size=0.3, random_state=42)


# In[46]:


linear_model = LinearRegression()


# In[48]:


linear_model.fit(X_train, y_train)


# # Question 13

# In[52]:



predictions = linear_model.predict(X_test)
rsquared = metrics.r2_score(y_test, predictions)
print("R squared score: ", round(rsquared, 2))


# #  Question 14

# In[53]:


rss = np.sum(np.square(y_test - predictions))
print("Residual Sum of Squares (RSS): ", round(rss, 2))


# # Question 15

# In[54]:


rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
print("Root Mean Squared Error: ", round(rmse, 3))


# # Question 16

# In[55]:


coe_det = metrics.r2_score(y_test, predictions)
print("Coefficient of determination: ", round(coe_det, 2))


# # Question 17

# In[56]:


X = normalised_df.drop(columns=['Appliances'])
Y = normalised_df['Appliances']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

linear_model_2 = LinearRegression()
linear_model_2.fit(x_train, y_train)


# In[57]:


predictions_2 = linear_model_2.predict(x_test)


# In[58]:


def get_weights_df(model, feat, col_name):
    """
    This function returns the weight of every feature.
    
    Source: https://gist.github.com/HamoyeHQ/3fd7570f8bf390ee9b9f7f042271d9f9
    """
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', col_name]
    weights_df[col_name].round(3)
    return weights_df


# In[59]:


linear_model_weights = get_weights_df(linear_model_2, x_train, 'Linear_Model_Weight')


# In[60]:


linear_model_weights.sort_values(by='Linear_Model_Weight', ascending=True)


# # Question 18 

# In[63]:


ridge_model = Ridge(alpha=0.4)
ridge_model.fit(x_train, y_train)


# In[64]:


predict_ridge = ridge_model.predict(x_test)


# In[65]:


rmse_rdige = np.sqrt(metrics.mean_squared_error(y_test, predict_ridge))
rmse_linear = np.sqrt(metrics.mean_squared_error(y_test, predictions_2))

print("Root Mean Squared Error (Ridge): ", round(rmse_rdige, 3))
print("Root Mean Squared Error (Linear): ", round(rmse_linear, 3))


# # Question 19

# In[83]:




lasso_model = Lasso(alpha=0.001)
lasso_model.fit(x_train, y_train)


# In[84]:


lasso_weights = get_weights_df(lasso_model, x_train, 'Lasso_Model_Weight')


# In[69]:


answer = lasso_weights[lasso_weights['Lasso_Model_Weight'] != 0]
answer


# In[70]:


len(answer)


# # Question 20

# In[79]:


lasso_predict = lasso_model.predict(x_test)
rmse_lasso = np.sqrt(metrics.mean_squared_error(y_test, lasso_predict))

print("Root Mean Squared Error (Lasso): ", round(rmse_rdige, 3))


# In[ ]:




