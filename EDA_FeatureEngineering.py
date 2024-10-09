#!/usr/bin/env python
# coding: utf-8

# # EDA and Fetaure Engineering,
# Data Cleaning and preparing the data for model training
# 

# In[129]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Problem Statement
# A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
# Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.
# 

# In[130]:


#importing the training dataset
df_train=pd.read_csv(r"C:\Users\Krish Ankola\Downloads\archive (6)\train.csv")
df_train.head()


# In[131]:


#importing the test dataset
df_test=pd.read_csv(r"C:\Users\Krish Ankola\Downloads\archive (6)\test.csv")
df_test.head()


# In[132]:


# we will merge or append or concat ( append didnt worked cause it needs higher pandas version the train and test dataset to do preprocessing on the data , if we are only given a dataset then we use 
#train test split to first split the data set into two parts( 80-20 ) generally


# In[133]:


df=pd.concat([df_train,df_test],ignore_index=True)
df.head()



# In[134]:


df.info()


# In[135]:


df.describe()


# In[136]:


df.drop(['User_ID'],axis=1,inplace=True)
df


# In[137]:


#we need to change categorical value into numerical we can use get dummies but then we need to add that df into the current df 
# so we use map 
# handling categorical feture gender
df['Gender']=df['Gender'].map({'F':0,'M':1})
df


# In[138]:


df['Age'].unique()


# In[139]:


# handle categorical feature age
df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})
df


# In[140]:


'''##second technqiue
from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'species'.
df['Age']= label_encoder.fit_transform(df['Age'])
 
df['Age'].unique()'''


# In[141]:


#fixing categorical value city category
df_city=pd.get_dummies(df['City_Category'],drop_first=True)
df_city = df_city.astype(int)
df_city 


# In[142]:


df=pd.concat([df,df_city],axis=1)


# In[143]:


df.head()


# In[144]:


# i have dropped the city_catogory which was categorical data 


# In[145]:


df.head()


# In[146]:


df.isnull().sum()


# In[147]:


df['Product_Category_2'].unique()


# In[148]:


df['Product_Category_2'].value_counts()


# In[149]:


# replace missing vlaues with mode fr product_category_2
df['Product_Category_2'].mode()[0]


# In[150]:


df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[151]:


df['Product_Category_2'].isnull().sum()


# In[152]:


# replace missing values for product_category_3
df['Product_Category_3'].mode()[0]


# In[153]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[154]:


df['Product_Category_3'].isnull().sum()


# In[155]:


df.head()


# In[156]:


df.drop(['City_Category'],axis=1,inplace=True)


# In[157]:


df.head()


# In[159]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')


# In[160]:


df.head()


# In[161]:


#convert object type into integer
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)


# In[162]:


df.head()


# In[163]:


df.info()


# In[166]:


sns.barplot(x='Age',y='Purchase',hue='Gender',data=df)


# By the bar graph we can observe that purchase behaviour or capacity is same for all age groups but men has high percentage of purchased goods as comapred to women 

# In[167]:


#visualisation purchase vs occupation
sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)


# In[168]:


sns.barplot(x='Product_Category_1',y='Purchase',hue='Gender',data=df)


# In[169]:


sns.barplot(x='Product_Category_2',y='Purchase',hue='Gender',data=df)


# In[170]:


sns.barplot(x='Product_Category_3',y='Purchase',hue='Gender',data=df)


# Feature Scaling

# In[175]:


df_test=df[df['Purchase'].isnull()]


# In[177]:


df_train=df[~df['Purchase'].isnull()]


# In[218]:


X=df_train.drop('Purchase',axis=1)


# In[219]:


X.head()
X.info()


# In[220]:


X.shape


# In[221]:


y=df_train['Purchase']
y
y.shape


# In[222]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)


# In[223]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[224]:


#train the model iss ke baad se train karna hai


# In[ ]:




