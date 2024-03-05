#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### import required Libraries


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#load movie data set
movie_data = pd.read_csv('tmdb_5000_movies.csv')
movie_data.head()  


# In[4]:


#find the shape of the movie data set
movie_data.shape


# In[5]:


movie_data.columns


# In[6]:


movie_data['id'].nunique()


# In[7]:


#load credit data set
credit_data = pd.read_csv('tmdb_5000_credits.csv')
credit_data.head()


# In[8]:


#find the shape of credit data set
credit_data.shape


# In[9]:


#rename the column 'id' as 'movie_id' to merge the two data set on the basis of movie id
movie_data.rename(columns={'id':'movie_id'}, inplace = True)
movie_data.head()


# In[10]:


#merge the two data set and creat a new DataFrame
df = movie_data.merge(credit_data, on = 'movie_id')
df.head()


# In[11]:


#find the shape of new DataFrame
df.shape


# In[12]:


df.columns


# # Data Cleaning

# In[13]:


#select some columns which serve our purpose and drop the rest column
#useful columns = [genres, movie_id, keywords, title_x, overview, cast, crew]


# In[14]:


df = df[['movie_id','title_x', 'keywords','genres','overview', 'cast', 'crew']]


# In[15]:


df.rename(columns ={'title_x' : 'title'}, inplace=True)
df.head()


# In[16]:


df.shape


# In[17]:


#check Null values
df.isna().sum()


# In[18]:


#drop the na column
df.dropna(inplace = True)
df.shape


# In[19]:


df.isna().sum()


# In[20]:


#check duplicates
df.duplicated().sum()


# In[21]:


df['keywords'].iloc[0]


# In[22]:


df['genres'].iloc[0]


# In[23]:


df['crew'].iloc[0]


# In[24]:


#extract the name from every row of the keywords coulmn
import ast
def convert(obj):
    L =[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[25]:


df['genres'] = df['genres'].apply(convert)
df.head(2)


# In[26]:


df['keywords'] = df['keywords'].apply(convert)
df.head(2)


# In[27]:


df['cast'][0]


# In[28]:


def convert2(obj):
    L = []
    counter = 0;
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1    
        else:
            break
    return L
       
   
    


# In[29]:


df['cast'] = df['cast'].apply(convert2)
df.head(2)


# In[30]:


df['crew'][0]


# In[31]:


#fetch director from the crew
def convert3(obj):
    L = []
    for i in ast.literal_eval(obj):
        if(i['job'] == 'Director'):
            L.append(i['name'])
            break
    return L
    


# In[32]:


df['crew'] = df['crew'].apply(convert3)
df.head(2)


# In[33]:


#convert overview column in a list
df['overview'] = df['overview'].apply(lambda x: x.split())


# In[34]:


df['genres'] = df['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
df.head(2)


# In[35]:


df['keywords'] = df['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[36]:


df['cast'] = df['cast'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[37]:


df['crew'] = df['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[38]:


df.head()


# In[39]:


#create a new column of tag by concatenating cast, crew, overview, genres and kewwords
df['tag'] = df['overview'] + df['genres'] + df['cast'] + df['crew'] + df['keywords']
df.head(2)


# In[40]:


new_df = df[['movie_id', 'title', 'tag']]
new_df.head(3)


# In[41]:


new_df['tag'] = new_df['tag'].apply(lambda x: " ".join(x))
new_df.head(2)


# In[42]:


new_df['tag'] = new_df['tag'].apply(lambda x:x.lower())
new_df.head(2)


# In[43]:


#apply stemming to remove same words appear in  different form
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[44]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[45]:


new_df['tag'] = new_df['tag'].apply(stem)


# In[46]:


new_df['tag'][0]


# In[69]:


# Now I have done text vectorization
#import countVectorizer from sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv =  CountVectorizer(max_features = 4800, stop_words ='english')
vectors = cv.fit_transform(new_df['tag'])
vectors.todense()


# In[70]:


cv.get_feature_names()


# In[71]:


from sklearn.metrics.pairwise import cosine_similarity


# In[77]:


similarity = cosine_similarity(vectors)
similarity


# In[73]:


similarity.shape


# In[55]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse= True, key= lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
                        


# In[78]:


recommend('Avatar')


# In[79]:


recommend('Batman Begins')


# In[ ]:




