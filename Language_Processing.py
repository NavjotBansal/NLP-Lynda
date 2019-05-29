#!/usr/bin/env python
# coding: utf-8

# # The Preprocessing part

# In[1]:


from nltk.corpus import stopwords
import pandas as pd
from string import punctuation
import re


# In[2]:


pd.set_option('max_colwidth',100)
data = pd.read_csv('SMSSpamCollection.tsv',sep='\t')


# ## No punctucation part

# In[3]:


#print(data)
data.columns=['label','bodytext']
#print(data)
#print(punctuation)
def no_punc(text):
    no_punc_text ="".join([char for char in text if char not in punctuation])
    return no_punc_text
data['body_text_nopunc']=data['bodytext'].apply(lambda x:no_punc(x))


# ## No spaces or breaks

# In[4]:


#print(data)
#print(stopwords.words('english'))
data['tokenized']=data['body_text_nopunc'].apply(lambda x:re.findall('\w+',x.lower()))


# ## No Stop Words

# In[5]:


def removestop(x):
    stop=[words for words in x if words not in stopwords.words('english')]
    return stop
data['token-nopunc-nostop']=data['tokenized'].apply(lambda x:removestop(x))


# # Working on the Chopping part

# ## Porter Stemming

# In[6]:


import nltk
ps = nltk.PorterStemmer()
#print(dir(wl))
#dir(ps)
# sad part about porter stemming is 
print(ps.stem('meaning'))
print(ps.stem('meanness'))
# check once for reference


# In[7]:


def stemtext(text):
    stemmed = [ps.stem(words) for words in text]
    return stemmed
data['body_stemmed']=data['token-nopunc-nostop'].apply(lambda x: stemtext(x))
#data


# ## Lemmatizing

# In[8]:


wl = nltk.WordNetLemmatizer()
#rint(wl.lemmatize('meanness'))
#rint(wl.lemmatize('meaning'))
#rint(wl.lemmatize('meanings'))
def lematizetext(text):
    lt = [ wl.lemmatize(words) for words in text ]
    return lt
data['body_lemmatized']=data['token-nopunc-nostop'].apply(lambda x: lematizetext(x))
data


# # Vectorization

# ## Count Vectorization

# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(analyzer=lematizetext)
x_train = vector.fit_transform(data['bodytext'])


# In[13]:


x_train_frame = pd.DataFrame(x_train.toarray())
#x_train_frame


# ## N-Gram 

# In[15]:


def cleantext(text):
    lt = " ".join([ wl.lemmatize(words) for words in text ])
    return lt
data['clean']=data['token-nopunc-nostop'].apply(lambda x: cleantext(x))


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
ngram = CountVectorizer(ngram_range=(2,2))
x_ngram_train = ngram.fit_transform(data['clean'])
#x_ngram_train
x_ngram_dframe = pd.DataFrame(x_ngram_train.toarray())
x_ngram_dframe.columns = ngram.get_feature_names()
#x_ngram_dframe


# ## TF-IDF 
# #### W(i,j) = TF(i,j)* log(N / DF(i))

# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer = lematizetext)
x_train_tfidf = tfidf.fit_transform(data['bodytext'])
x_tfidf_dataframe = pd.DataFrame(x_train_tfidf.toarray())
x_tfidf_dataframe.columns = tfidf.get_feature_names()
#x_tfidf_dataframe


# # Feature Extraction

# In[36]:


data['totalwords']=data['bodytext'].apply(lambda x: len(x) - x.count(" "))
def punc(x):
    count = sum([1 for char in x if char in punctuation])/(len(x)-x.count(" "))
    count = round(count*100,3)
    return count
data['punctutaion']=data['bodytext'].apply(lambda x: punc(x))
data


# In[37]:


from matplotlib import pyplot
import numpy as np


# In[41]:


bins =np.linspace(0,200,40)
pyplot.hist(data[data['label']=='ham']['totalwords'],bins,alpha=0.6,normed=True,label='ham')
pyplot.hist(data[data['label']=='spam']['totalwords'],bins,alpha=0.6,normed=True,label='spam')
pyplot.legend(loc='upper left')
pyplot.show()


# In[46]:


bins = np.linspace(0,10,40)
pyplot.hist(data[data['label']=='ham']['punctutaion'],bins,alpha=0.6,normed=True,label='ham')
pyplot.hist(data[data['label']=='spam']['punctutaion'],bins,alpha=0.6,normed=True,label='spam')
pyplot.legend(loc='upper left')
pyplot.show()


# In[ ]:




