#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


csv=pd.read_csv("C:\\Users\\anshu\\Downloads\\archive (5)\\Advertising.csv")
csv.info()


# In[3]:


csv.head(10)


# In[6]:


csv.info()


# In[7]:


csv.drop("Unnamed: 0",axis=1,inplace=True)


# In[9]:


csv.isnull().sum()


# In[8]:


sns.scatterplot(x="TV",y="Sales",data=csv)


# In[11]:


sns.scatterplot(x="Radio",y="Sales",data=csv)


# In[12]:


sns.scatterplot(x="Newspaper",y="Sales",data=csv)


# In[10]:


sns.pairplot(csv,hue='Sales')


# In[13]:


from sklearn.preprocessing import StandardScaler
y = csv['Sales']
scaler = StandardScaler()
x = scaler.fit_transform(csv.drop(columns=["Sales"]))
x = pd.DataFrame(data=x, columns=csv.drop(columns=["Sales"]).columns)


# In[14]:


k=x.corr()


# In[15]:


z = [(str(k.columns[i]), str(k.columns[j])) for i in range(len(k.columns)) for j in range(i+1, len(k.columns)) if abs(k.corr().iloc[i, j]) > 0.5]


# In[16]:


z,len(z)


# In[17]:


x


# In[18]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = x
vif = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])], index=vif_data.columns)
vif


# In[19]:


def remover(csv):
    vif = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])], index=vif_data.columns)
    if vif.max() == float('inf') or vif.max()>68:
        column_to_drop = vif[vif == vif.max()].index[0]
        print(vif[vif == vif.max()].index[0])
        csv=csv.drop(columns=[column_to_drop])
    else:
        pass
    return csv
for i in range(50):
    vif_data=remover(vif_data)


# In[20]:


vif_data.head(10)


# In[21]:


X=vif_data
y=csv['Sales']


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


from sklearn.linear_model import LinearRegression
logmodel = LinearRegression()

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)


# In[24]:


predictions


# In[25]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

predictions = logmodel.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)  # RMSE
r_squared = r2_score(y_test, predictions)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r_squared)


# In[26]:


result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
result_df


# In[ ]:




