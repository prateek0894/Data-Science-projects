# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:45:12 2020

@author: Prateek Gupta
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

#reading the train and test dataset
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train1=train
test1=test

#Exploring Train dataset
train.head(10)
train.describe()

#Converting Categorical variables into numerical variables using Label encoding or by creating dummy variables based on whether variables  are ordinal or nominal
labelencoder_X = LabelEncoder()
train1['Gender'] = labelencoder_X.fit_transform(train1['Gender'])
train1['Age'] = labelencoder_X.fit_transform(train1['Age'])
train1['Stay_In_Current_City_Years'] = labelencoder_X.fit_transform(train1['Stay_In_Current_City_Years'])


p1=pd.get_dummies(train1["City_Category"])    

train1=pd.concat([train1,p1],axis=1)

#removing the variables not required
train1.drop(["User_ID","Product_ID","City_Category"], axis=1,inplace =True)

#imputation applied on missing values
train1.isna().sum()

train1['Product_Category_2']=train1['Product_Category_2'].fillna(0)
train1['Product_Category_3']=train1['Product_Category_3'].fillna(0)

#separating independent and dependent variables
x=train1
y=train1["Purchase"]
x.drop("Purchase",axis=1,inplace=True)

#splitting the dataset and into train and test dataset and scaling them laterwards
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.50,random_state =205)
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_val = sc_X.transform(x_val)

#Applying same changes on test dataset
test['Gender'] = labelencoder_X.fit_transform(test['Gender'])
test['Age'] = labelencoder_X.fit_transform(test['Age'])
test['Stay_In_Current_City_Years'] = labelencoder_X.fit_transform(test['Stay_In_Current_City_Years'])
t1=pd.get_dummies(test["City_Category"])    

test=pd.concat([test,t1],axis=1)

test.drop(["User_ID","Product_ID","City_Category"], axis=1,inplace =True)

test.isna().sum()

test['Product_Category_2']=test['Product_Category_2'].fillna(0)
test['Product_Category_3']=test['Product_Category_3'].fillna(0)

test= sc_X.transform(test)

#training the model

#######decision tree regression
######################
from sklearn.tree import DecisionTreeRegressor

#creating and fitting the model
regressor1=DecisionTreeRegressor(random_state=0,max_depth=15,max_leaf_nodes=250,max_features=10,min_samples_leaf=4, min_samples_split=6,)
regressor1.fit(x_train,y_train)

#predicting the model on both splitted test and train dataset in order to test overfitting
pred1=regressor1.predict(x_val)
trainpred1=regressor1.predict(x_train)

#Checking accuracy with the help of root mean square error
from math import sqrt
rmstest= sqrt(mean_squared_error(y_val,pred1)) 
print(rmstest)
#rmstest=2939.9065296589233
rmstrain= sqrt(mean_squared_error(y_train,trainpred1))
print(rmstrain)
#rmstrain=2921.3252916850715

#NOTE: Since rmstest and rmstrain are almost equal, thus we can say that there is no overfitting in the model

############################
####### Random Forest Regression
############################

from sklearn.ensemble import RandomForestRegressor

#creating and fitting the model
regressor2=RandomForestRegressor(random_state=0,max_depth=15, max_features=10, max_leaf_nodes=250,min_samples_leaf=1, min_samples_split=10,n_estimators=150)
regressor2.fit(x_train,y_train)

#predicting the model on both splitted test and train dataset in order to test overfitting
pred2=regressor2.predict(x_val)
trainpred2=regressor2.predict(x_train)

#Checking accuracy with the help of root mean square error
rmstest2= sqrt(mean_squared_error(y_val,pred2))
print(rmstest2)
#rmstest=2928.510860471679

rmstrain2= sqrt(mean_squared_error(y_train,trainpred2))
print(rmstrain2)
#rmstrain=2907.5582314718363

#NOTE: Since rmstest and rmstrain are almost equal, thus we can say that there is no overfitting in the model

#since we found that Random forest regresion gives better rms score, thus I will use regressor2 model to do final prediction and make submission.

predtest=regressor2.predict(test)
submission = pd.DataFrame({"Purchase":predtest,'User_ID': test1['User_ID'], 'Product_ID': test1['Product_ID'] })
submission.to_csv("submission_black_friday_sales.csv",index=False)





