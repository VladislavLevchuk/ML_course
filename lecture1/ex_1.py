#!/usr/bin/env python
# coding: utf-8

# # Numpy

# ## Part 1)

# In[1]:


import numpy as np


# In[2]:


# Create array "a"
a = np.array([[1,2,3,3,1], [6,8,11,10,7]]).T
print(a)


# In[3]:


# Find mean of array column-wise
mean_a = np.mean(a, axis=0)
print(mean_a)


# ## Part 2)

# In[4]:


# Demean array "a"
a_centered = (a - mean_a)
print(a_centered)


# ## Part 3)

# In[5]:


# Find scalar product of two columns
a_centered_sp = np.dot(a_centered[:,0],a_centered[:,1])
print(a_centered_sp)


# In[6]:


# Devide scalar product by the (# of observations N - 1): covariance of two columns
N = len(a)
cov = a_centered_sp / (N-1)
print(cov)


# ## Part 4)

# In[7]:


# Find covariance using built-in method
cov_alt = np.cov(a.T)
print(cov_alt[0,1])


# # Pandas

# ## Part 1)

# Create DFs

# In[8]:


import pandas as pd


# In[9]:


authors = pd.DataFrame({'author_id':[1, 2, 3], 'author_name':['Тургенев', 'Чехов', 'Островский']})
authors


# In[10]:


book = pd.DataFrame({'author_id':[1, 1, 1, 2, 2, 3, 3], 
                     'book_title':['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'], 
                     'price':[450, 300, 350, 500, 450, 370, 290]})
book


# ## Part 2)

# Merge DFs

# In[11]:


authors_price = pd.merge(left=authors, right=book, how='inner', on='author_id')
authors_price


# ## Part 3)

# Find 5 most expensive books

# In[12]:


top5 = authors_price.sort_values(by='price', ascending=False).reset_index().loc[:4,['book_title','price']]


# ## Part 4)

# Calculate ststistics for authors

# In[13]:


authors_stat = authors_price.groupby('author_name')['price'].agg(['min', 'max', 'mean']).reset_index()
authors_stat.columns = ['author_name', 'min_price', 'max_price', 'mean_price']

round(authors_stat,2)


# ## Part 5)

# Add new column 'cover'

# In[14]:


authors_price['cover'] = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
authors_price


# Calculate statistics for summed cover prices for each author and cover type using Pivot Table

# In[15]:


book_info = pd.pivot_table(authors_price, values='price', index=['author_name'],
                                                    columns=['cover'], aggfunc=np.sum)
book_info = book_info.fillna(0)
book_info


# Save the resulting table as pickle

# In[16]:


book_info.to_pickle("book_info.pkl")


# In[17]:


book_info2 = pd.read_pickle("book_info.pkl")
book_info2

