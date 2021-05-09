#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import sys

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
#from nltk.corpus import stopwords


# In[2]:


class TextProcessing(object):
    def __init__(self,train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        
    def filter_df(self, df, function):
        return df.apply(function)
    
    def tokenize(self):
        def split_words(input_text):
            words = input_text.split()
            return words
        self.train_set = self.filter_df(self.train_set, split_words)
        self.test_set = self.filter_df(self.test_set, split_words)        
        
        #df = pd.read_csv("train.tsv", sep='\t')
        #df['Review'] = df['Review'].str.upper()  #turn to upper case
    def uppercase(self):
        #def set_to_uppercase(words):
        #    transformed_list = [i.upper() for i in words]
         #   return transformed_list
        #self.train_set = self.filter_df(self.train_set, set_to_uppercase)
        #self.test_set = self.filter_df(self.test_set, set_to_uppercase)
        def set_to_lowercase(words):
            transformed_list = [i.lower() for i in words]
            return transformed_list
        self.train_set = self.filter_df(self.train_set, set_to_lowercase)
        self.test_set = self.filter_df(self.test_set, set_to_lowercase)
            
        
    def remove_other(self):
        def remove_junk(words):
            transformed_list=[]
            for i in words:
                i = i.replace('[^\w\s]','')
                transformed_list.append(i)
            return transformed_list    
        self.train_set = self.filter_df(self.train_set, remove_junk)
        self.test_set = self.filter_df(self.test_set, remove_junk)
            
        #df['Review'] = df['Review'].str.replace('[^\w\s]','')  #remove whitespaces and punctuation
        #df.head(20)
    
        #sabiranje svih reci da se vidi koje se najvise ponavljaju a nemaju nekog znacaja
        #df.Review.str.split(expand=True).stack().value_count[:50]  #tokenize

    def remove_stop_words(self):
        def remove_stopwords(input_text):
            stopwords = ['JE','I', 'DA', 'SVE', 'ZA', 'U', 'NA', 'SU', 'SAM', 'SE', 'OD', 'A', 'ALI', 'SA', 'SMO']
            transformed_list=[i for i in input_text if i not in stopwords]
            return transformed_list
        self.train_set = self.filter_df(self.train_set, remove_stopwords)
        self.test_set = self.filter_df(self.test_set, remove_stopwords)
        
    def detokenize(self, dataset):
        return dataset.apply(lambda x: ''.join(i + ' ' for i in x))

    def pipe(self):
        self.tokenize()
        self.uppercase()
        self.remove_other()
        self.remove_stop_words()


# In[3]:


def read_data(train_data_path, test_data_path):
    train = pd.read_csv("train.tsv", sep='\t')
    test = pd.read_csv("test_preview.tsv", sep='\t')
    
    X_train, Y_train, X_test, Y_test = train.Review, train.Sentiment, test.Review, test.Sentiment
    return X_train, Y_train, X_test, Y_test


# In[4]:


if __name__ == "__main__":
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    
    X_train, Y_train, X_test, Y_test = read_data(train_data_path, test_data_path)
    
    processed_text = TextProcessing(X_train, X_test)
    processed_text.pipe()
    
    X_train, X_test = processed_text.train_set, processed_text.test_set
    X_train = processed_text.detokenize(X_train)
    X_test = processed_text.detokenize(X_test)
    
    #tfid vectorizer
    tf_id_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
    train_X =  tf_id_vectorizer.fit_transform(X_train).toarray()
    test_X =  tf_id_vectorizer.transform(X_test).toarray()
    
    linear_SVM = LinearSVC(C=0.325, fit_intercept=False)
    linear_SVM.fit(train_X, Y_train)
    y_pred = linear_SVM.predict(test_X)

    print(accuracy_score(Y_test, y_pred))


    
    
    
    


# In[ ]:




