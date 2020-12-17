#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


# In[89]:


data = pd.read_csv('search_clustering.csv', engine = 'python')
# feature engineering & extraction

data = data.filter(items=['pubmed_id', 'title', 'abstract', 'copyrights', 'publication_date', 'authors', 'keywords', 'journal', 'conclusions', 'methods', 'results', 'Rank', 'Total Cites', 'Journal Impact Factor', 'Eigenfactor Score', 'published_year', 'published_month'])
data = data.dropna(subset=['Journal Impact Factor'])

data.head()


# In[90]:


data.dtypes


# In[91]:


data['publication_date'] = pd.to_datetime(data.publication_date, errors ='coerce')
data['published_year'] = pd.DatetimeIndex(data.publication_date).year
data['published_month'] = pd.DatetimeIndex(data.publication_date).month


# In[92]:


data.head()


# In[93]:


df_fit = data.filter(items=['pubmed_id','published_year', 'published_month', 'journal', 'Journal Impact Factor', 'Eigenfactor Score'])


# In[94]:


df_fit.shape


# In[95]:


import re

for index, row in df_fit.iterrows():
    if re.match(r'^-?\d+(?:\.\d+)?$', row['Journal Impact Factor'].strip()) is None:
        print(row['Journal Impact Factor'])
        df_fit.drop(index, inplace=True)
        data.drop(index, inplace=True)

df_fit


# In[96]:


data.shape


# In[97]:


# df_fit["ranking_score"] = float(df_fit["published_year"].values)*0.3/1000 \
#                             + float(df_fit["published_month"].values)*0.2/10 \
#                             + float(df_fit["Journal Impact Factor"].values)*0.5

a = df_fit["published_year"].values *0.3/1000
b = df_fit["published_month"].values *0.2/10
c = df_fit["Journal Impact Factor"].values
c = [float(x)* 0.5 for x in c] 

df_fit["ranking_score"] = a + b + c

df_fit.sort_values("ranking_score", ascending=False).reset_index(drop=True)


# In[98]:


df_fit = pd.get_dummies(df_fit, columns=['journal'], drop_first = True)


# In[99]:


df_fit["Eigenfactor Score"].fillna(0, inplace = True) 
df_fit["Journal Impact Factor"].fillna(0, inplace = True)  
df_fit["ranking_score"].fillna(0, inplace = True)  


# In[100]:


wcss = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df_fit.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,5))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()


# In[103]:


kmeans = KMeans(n_clusters=5, random_state=42).fit(df_fit.iloc[:,1:]) # use 5 as cluster number hyperparameter
df_fit["label"] = kmeans.predict(df_fit.iloc[:,1:])
data["label"] = kmeans.predict(df_fit.iloc[:,1:])


# In[104]:


data.head()


# In[105]:


def return_label(df, df1):
    articles = []
    label = df.iloc[0]["label"]
    for i,row in df1.iterrows():
        if row["label"] == label:
            articles.append(row[["pubmed_id","title","journal"]])
    return articles[:10]


# In[106]:


a = return_label(df_fit, data)
df_export = pd.DataFrame(a)
export_csv = df_export.to_csv (r'recommend.csv', index = None, header=True) 


# In[ ]:




