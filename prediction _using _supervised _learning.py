#!/usr/bin/env python
# coding: utf-8

# Name : Merna alaa elden
# 
# Track : Data science & Business Analytics.
# 
# Task 1 :prediction the student score percentage using supervised learning (linear regression algorithm).

# In[76]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import io
from sklearn import metrics  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
import warnings
warnings.filterwarnings('ignore')


# In[55]:


df = pd.read_excel("D:\DATA SCIENCE\sparks\student_scores.xlsx" )
print(df)


# In[56]:


df.isnull().sum()


# In[57]:


df.shape


# In[58]:


df.columns


# In[70]:


X = df['Hours']
Y = df['Scores']
plt.plot( X, Y , 'o')
plt.title("Hours vs Scores")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[71]:


df.describe()


# In[73]:


X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values  


# In[75]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10) 


# In[77]:


regression = LinearRegression()  
regression.fit(X_train, Y_train) 


# In[80]:


line = regression.coef_*X+regression.intercept_
plt.scatter(X, Y)
plt.plot(X, line)
plt.show()


# In[83]:


pred = regression.predict(X_test)
print("predicted value" , pred)


# In[85]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': pred})  
df


# What will be predicted score if the student studies for 9.25 hrs/day?

# In[88]:


hours = [[9.25]]
test_pred = regression.predict(hours)
print("number of hours= {}".format(hours))
print("predicted score ={}".format(test_pred[0]))


# In[89]:


#evalute the model to test the performance of the algorithm
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_test, pred)) 


# In[ ]:




