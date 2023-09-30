import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('Admission_Predict_Ver1.1.csv')
data.shape
data.head()
data.tail()

data.info()
data.describe()
data.isna().sum()


data1=pd.read_csv('Admission_Predict_Ver1.1.csv',na_values=[0])

data1.isna().sum()

columns=['GRE Score','TOEFL Score','University Rating','SOP','CGPA']

for column in columns:
    mean=data[column].mean()
    data[column]=data[column].replace(0,mean)

###################
plt.figure(figsize=(10,6))
sns.pairplot(data)

data.columns
columns=['GRE Score','TOEFL','University Rating','SOP','LOR','CGPA']
for column in columns :
    plt.figure(figsize=(10,8))
    sns.boxplot(x=data["Chance of Admit "],y=data[column])
    print('\n')

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)

x=data.drop(['Chance of Admit'],axis=1)
y=data['Chance of Admit']