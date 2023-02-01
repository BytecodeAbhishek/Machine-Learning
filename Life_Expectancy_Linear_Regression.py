#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


# In[2]:


df = pd.read_csv("C:/Users/DITU/Desktop/Life Expectancy Data.csv")
df.head()


# In[6]:


# to drop out null values
df = df.dropna() 
x = df.iloc[:,16:]
x = x.drop("Population",axis=1)
y = df.iloc[:,3]
y


# In[4]:


# plt.scatter(x,y,color='green')
# plt.show()
plt.plot(x,y)


# In[7]:


from sklearn.linear_model import LinearRegression
x_train , x_test, y_train,y_test=train_test_split(x,y,test_size=0.12)
model = LinearRegression().fit(x_train, y_train)


# In[8]:


model_confidence = model.score(x_test,y_test)
print(model_confidence)


# In[ ]:




