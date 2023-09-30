import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#step2
data=pd.read_csv('diabetes.csv')
############################
data.head()
data.tail()
data.info()
pd.set_option('display.max_columns',50)
data.describe()

data.columns
data.isna().sum()
##############################
data1=pd.read_csv('diabetes.csv',na_values=[0])

data1.isna().sum()

columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']

for column in columns:
    mean=data[column].mean()
    data[column]=data[column].replace(0,mean)

#############################
#data['Glucose'].mean()
#data['Glucose'].replace(0,mean)
###############################

plt.figure(figsize=(12,8))
sns.pairplot(data)

data.columns
columns=['Pregnancies','Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age']
for column in columns :
    plt.figure(figsize=(10,8))
    sns.boxplot(x=data["Outcome"],y=data[column])
    print('\n')
    
##########################################
    
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)

###########################################

#seggregate input and output
x=data.drop(['Outcome'],axis=1)
y=data['Outcome']

################################################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=0)

####################################################
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

#####################################################
y_pred=classifier.predict(x_test)
####################################################
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
#########################################

y_pred=classifier.predict(x_test)
##############################################

from sklearn import metrics

metrics.confusion_matrix(y_test, y_pred)
metrics.accuracy_score(y_test, y_pred)

print(metrics.classification_report(y_test,y_pred))