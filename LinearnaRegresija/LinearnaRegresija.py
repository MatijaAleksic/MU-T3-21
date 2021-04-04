#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import csv

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


# #Iscrtavanje linearne regresije
#df = pd.read_csv("train.csv")

# %matplotlib inline
# plt.xlabel('Width')
# plt.ylabel('Weight')
# plt.scatter(df.Width,df.Weight, color='red', marker='+')


# In[3]:


#Normalizacija podataka
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# df = normalize(df)
# %matplotlib inline
# plt.xlabel('Weight')
# plt.ylabel('Width')
# plt.scatter(df.Width, df.Weight, color='red', marker='+')


# In[4]:


# #Boxplot da se vide outlieri
# df.plot(kind='box')
# plt.show()

# df.describe()


# In[5]:


#PRONAZENJE PARAMETARA ZA OUTLIERE
def up_down(df):
    Q1 = df.describe().loc['25%']
    Q3 = df.describe().loc['75%']

    IQR_Weight = Q3["Weight"] - Q1['Weight']
    IQR_Width = Q3["Width"] - Q1['Width']

    #Racunanje whiskera gore i dole
    UP_Weigth = Q3["Weight"] + (1.5 *  IQR_Weight)
    DOWN_Weigth = Q1["Weight"] - (1.5 *  IQR_Weight)

    UP_Width = Q3["Width"] + (1.5 *  IQR_Width)
    DOWN_Width = Q1["Width"] - (1.5 *  IQR_Width)

    return [UP_Weigth,DOWN_Weigth,UP_Width,DOWN_Width]


# In[6]:


def up_down_z(df, sd):    
    STD = df.describe().loc['std']
    STDw = STD['Weight']
    STDwi = STD['Width']

    MEAN = df.describe().loc['mean']
    MEANw = MEAN['Weight']
    MEANwi = MEAN['Width']

    up_w = MEANw + sd * STDw;
    down_w = MEANw - sd * STDw;

    up_wi = MEANwi + sd * STDwi;
    down_wi = MEANwi - sd * STDwi;

    return [up_w,down_w,up_wi,down_wi]


# In[7]:


def takeout_outliers(df, up_w, down_w, up_wi, down_wi):
    indexes = []
    counter = 0
    for i in df.Weight:
        if i > up_w or i < down_w:
            indexes.append(counter)
        counter += 1

    counter = 0
    for i in df.Width:
        if i > up_wi or i < down_wi:
            indexes.append(counter)
        counter += 1
    
    data = df.drop(indexes)
    return data


# In[8]:


# #IZBACIVANJE OUTLIERA
# [up_w,down_w,up_wi,down_wi] = up_down_z(df, 1)
# #[up_w,down_w,up_wi,down_wi] = up_down(df)

# df = takeout_outliers(df,up_w,down_w,up_wi,down_wi)

# %matplotlib inline
# plt.xlabel('Width')
# plt.ylabel('Weight')
# plt.scatter(df.Width,df.Weight, color='red', marker='+')


# In[9]:


def nadji_hiperparametre(df):
    #Pronalazenje x meana
    means = df.describe().loc['mean']
    x_mean = means['Width']
    y_mean = means['Weight']
    
    #Broj elemenata
    n = df.Width.size

    #pravljenje x - x_mean liste
    x_xmean = []
    for x in df.Width:
        x_xmean.append(x - x_mean)

    #pravljenje y - y_mean liste
    y_ymean = []
    for y in df.Weight:
        y_ymean.append(y - y_mean)


    #racunanje sume od (x - x_mean) * (y - y_mean)
    sum_xxmean_yymean = 0
    for i in range(len(x_xmean)):
        mult = x_xmean[i] * y_ymean[i]
        sum_xxmean_yymean += mult

    #print("Suma (x - x_mean) * (y - y_mean) = " + str(sum_xxmean_yymean))


    #racunanje sume (x - x_mean)^2
    x_kvad_suma = 0
    for x in x_xmean:
        x_kvad_suma += x * x

    #racunanje sume (y - y_mean)^2
    y_kvad_suma = 0
    for y in y_ymean:
        y_kvad_suma += y * y

#    print("Suma (x - x_mean)^2 = " + str(x_kvad_suma))
#    print("Suma (y - y_mean)^2 = " + str(y_kvad_suma))

    #racunaje vrijednosti r
    r = sum_xxmean_yymean / np.sqrt(x_kvad_suma * y_kvad_suma)
#    print("Vrijednost r = " + str(r))


    Sy = np.sqrt(y_kvad_suma/(n-1))
#    print("Vrijednost Sy = " + str(Sy))

    Sx = np.sqrt(x_kvad_suma/(n-1))
#    print("Vrijednost Sx = " + str(Sx))

    b = r * (Sy / Sx)
#    print("Vrijednost b = " + str(b))

    a = y_mean - b * x_mean
#    print("Vrijednost a = " + str(a))


    #print("\n Funkcija izgleda ovako: \n y =" + str(b) + " * x + " + str(a))
    
    return [a,b]


# In[10]:


def gradient_descent(df):
    X = df.iloc[:,1]
    Y = df.iloc[:,0]
    
    m = 0
    c = 0
    
    L = 0.0001
    
    n = df.Weight.size
    
    for i in range(len(X)):
        X_pred = m* Y + c
        D_m = (-2/n) * sum(Y * (X - X_pred))
        D_c = (-2/n) * sum(X - X_pred)
        m = m - L * D_m
        c = c - L * D_c
        
    return [c,m]


# In[11]:


# #ISCRTAVANJE LINIJE LINEARNE REGRESIJE
# hiperparametri = nadji_hiperparametre(df)
# #hiperparametri = gradient_descent(df)
# a = hiperparametri[0]
# b = hiperparametri[1]

# y = []
# for x in df.Width:
#     y.append(b * x + a)
    
# plt.scatter(df.Width, df.Weight, color='red', marker='+')
# plt.plot(df.Width,y, color='green',linewidth=2, markersize=12)


# In[12]:


def takeout_long_residuals(df, takeout_distance):
   indexes = []
   distances = []
   counter = 0

   for index, row in df.iterrows():
       x2 = row['Width']
       y2 = row['Weight']

       c = y2 + b * x2
       x1 = (c - a)/ (2* b)
       y1 = b * x1 + a
       dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
       if dist > takeout_distance:
           distances.append(dist)
           indexes.append(index)

   data = df.drop(labels=indexes, axis=0)
   return data

# df = takeout_long_residuals(df, 185)
# #distances
# #df


# y = []
# for x in df.Weight:
#     y.append(a + b * x)
   
# plt.scatter(df.Width, df.Weight, color='red', marker='+')
# plt.plot(y, df.Weight, color='green',linewidth=2, markersize=12)


# In[13]:


#Racunanje RMSE
def RMSE(n,predicted, acctual):
    suma = 0
    for i in range(0, n):
        suma += (predicted[i] - acctual.iloc[i])**2
    return np.sqrt(suma/n)
        


# In[14]:


import sys

if(len(sys.argv) != 3):
    print("Mora imati dva argumenta 'tran.csv' 'test.csv'")
    exit()
else:
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]   
    
    df = pd.read_csv(train_csv)

    #[up_w,down_w,up_wi,down_wi] = up_down_z(df, 0.9)
    [up_w,down_w,up_wi,down_wi] = up_down(df)

    df = takeout_outliers(df,up_w,down_w,up_wi,down_wi)

    hiperparametri = nadji_hiperparametre(df)
    #hiperparametri = gradient_descent(df,1000)

    a = hiperparametri[0]
    b = hiperparametri[1]

    df_test = pd.read_csv(test_csv)
    y = []
    for x in df_test.Width:
        y.append(b * x + a)
        
    n = df_test.Weight.size
    print(RMSE(n, y, df_test.Weight))
    
#     plt.scatter(df.Width, df.Weight, color='red', marker='+')
#     plt.plot(df.Width,y, color='green',linewidth=2, markersize=12)





