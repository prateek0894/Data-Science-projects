# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:11:04 2020

@author: Prateek Gupta
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from  sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train.head(10)
train.describe()

#imputation
train['education']=train['education'].fillna("Bachelor's")
train['previous_year_rating']=train['previous_year_rating'].fillna(3)

train1=train

#dummy variables
p1=pd.get_dummies(train1["department"])    
p2=pd.get_dummies(train1["education"])
p3=pd.get_dummies(train1["recruitment_channel"])

labelencoder_X = LabelEncoder()
train1['gender'] = labelencoder_X.fit_transform(train1['gender'])

train1=pd.concat([train1,p1,p2,p3],axis=1)

#removing columns
train1.drop(["employee_id","department","region","education","recruitment_channel","region"], axis=1,inplace =True)

x=train1
x.drop("is_promoted",axis=1,inplace=True)
y=train["is_promoted"]

#splitting dataset
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.20,random_state =205)
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_val = sc_X.transform(x_val)

#test dataset
test['education']=test['education'].fillna("Bachelor's")
test['previous_year_rating']=test['previous_year_rating'].fillna(3)

test1=test

#dummy variables
t1=pd.get_dummies(test1["department"])    
t2=pd.get_dummies(test1["education"])
t3=pd.get_dummies(test1["recruitment_channel"])


test1['gender'] = labelencoder_X.fit_transform(test1['gender'])

test1=pd.concat([test1,t1,t2,t3],axis=1)


#removing columns
test1.drop(["employee_id","department","region","education","recruitment_channel","region"], axis=1,inplace =True)

test1 = sc_X.fit_transform(test1)


#implementing classifier models

#####Logistic regression
################################

from sklearn.linear_model import LogisticRegression

classifier1=LogisticRegression(random_state=0)
classifier1.fit(x_train,y_train)

pred1=classifier1.predict(x_val)
trainpred1=classifier1.predict(x_train)

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_val,pred1)
print(cm1)

from sklearn.metrics import f1_score

f1_score(y_val,pred1, average='binary')
f1_score(y_train,trainpred1, average='binary')

#####Decision Tree Classification
################################

from sklearn.tree import DecisionTreeClassifier

classifier2=DecisionTreeClassifier(criterion="gini",random_state=0,max_leaf_nodes=100,max_features=15)
classifier2.fit(x_train,y_train)

pred2=classifier2.predict(x_val)
trainpred2=classifier2.predict(x_train)

cm2=confusion_matrix(y_val,pred2)
print(cm2)

from sklearn.metrics import f1_score

f1_score(y_val,pred2, average='binary')
f1_score(y_train,trainpred2, average='binary')


#####Random Forest Classification
################################

from sklearn.ensemble import RandomForestClassifier

classifier3=RandomForestClassifier(n_estimators=1000, criterion='gini', random_state=0, max_depth=100,max_features=22, min_samples_leaf=4, min_samples_split=5,max_leaf_nodes=100)
classifier3.fit(x_train,y_train)

pred3=classifier3.predict(x_val)
trainpred3=classifier3.predict(x_train)

cm3=confusion_matrix(y_val,pred3)
print(cm3)

from sklearn.metrics import f1_score

f1_score(y_val,pred3, average='binary')
#f1 score=
f1_score(y_train,trainpred3, average='binary')

#####Applying Artificial Neural Network
################################

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier4=Sequential()

classifier4.add(Dense(output_dim=10, init='uniform',activation='relu',input_dim=19))

classifier4.add(Dense(output_dim=10, init='uniform',activation='relu'))

classifier4.add(Dense(output_dim=1, init='uniform',activation='sigmoid'))

classifier4.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier4.fit(x_train, y_train,batch_size=10,nb_epoch=100)

pred4=classifier4.predict(x_val)

pred4=(pred4>0.30)

trainpred4=classifier4.predict(x_train)
trainpred4=(trainpred4>0.30)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm4=confusion_matrix(y_val,pred4)
print(cm4)

from sklearn.metrics import f1_score

f1_score(y_val,pred4, average='binary')
f1_score(y_train,trainpred4, average='binary')

#Applying ANN classification on final Test dataset as it gave the best f1 score
pred_test=classifier4.predict(test1)
pred_test=(pred_test>0.30)
submission = pd.DataFrame({'employee_id': test['employee_id'] })
submission["is_promoted"]=pred_test.astype(int)
submission.to_csv("HR_Analytics_submission.csv",index=False)



