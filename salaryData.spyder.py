#!pip instaii matplotlib
#step1: importin all pacakage 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#step2: import dataset
data=pd.read_csv('Salary_Data.csv')
#EDA(eexploratory data analysis)
data.shape #number of rows and columns

data.head()
data.head(10)
data.tail()

data.describe()


data.columns

data.info()
####################################################################
#corr:correlation between the variation
data.corr()
#heatmap: correlation graph
sns.heatmap(data.corr(),annot=True)
#to find null value 
data.isna()
data.isna().sum()
######################################################################
#step:seggregate inpuut and output parameters

x=data.iloc[:,0:1]
y=data.iloc[:,1:2]

#####################################################
#step5:split data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=0)

##################################################################
x_train.shape
x_test.shape
y_train.shape
y_test.shape
###############################################################

regressor.fit(x_train, y_train)

regressor.coef_
regressor.intercept_

#y=mx=c
#y=9312*+26780


###########
y_pred=regressor.predict(x_test)
y_pred
######################################################
from sklearn import matrics
np.sqrt(metrics.mean_squared_ery_testror(, y_pred))

metrics.mean_absolute_error(y_test, y_pred)

metrics.r2_score(y_test, y_pred)