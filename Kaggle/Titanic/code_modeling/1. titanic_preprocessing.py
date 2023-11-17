#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Titanic Preprocessing
training=pd.read_csv('C:/Users/82103/Desktop/프로젝트/Titanic/Titanic_data/train.csv')
test=pd.read_csv('C:/Users/82103/Desktop/프로젝트/Titanic/Titanic_data/test.csv')

training['train_test']=1
test['train_test']=0
test['Survived']=np.NaN
all_data=pd.concat([training, test])

df_num=training[['Age', 'SibSp', 'Parch', 'Fare']] # numeric variable
df_cat=training[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']] # categorical variable

pd.pivot_table(training, index='Survived', values=['Age', 'SibSp', 'Parch', 'Fare'])
pd.pivot_table(training, index='Survived', columns='Pclass', values='Ticket', aggfunc='count')
pd.pivot_table(training, index='Survived', columns='Sex', values='Ticket', aggfunc='count')
pd.pivot_table(training, index='Survived', columns='Embarked', values='Ticket', aggfunc='count')

# To see if anyone booked multiple cabins
all_data['cabin_multiple']=all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
training['cabin_multiple']=training.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
pd.pivot_table(training, index='Survived', columns='cabin_multiple', values='Ticket', aggfunc='count')

# See letters in cabin 
all_data['cabin_adv']=all_data.Cabin.apply(lambda x: str(x)[0])

# Understand ticket values
all_data['numeric_ticket']=all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters']=all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1])>0 else 0)

# Person's title ex) Mr. Mrs. etc
all_data['name_title']=all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

all_data.Age=all_data.Age.fillna(training.Age.median())
all_data.Fare=all_data.Fare.fillna(training.Fare.median())

all_data.dropna(subset=['Embarked'], inplace=True)

# Normalize data
all_data['norm_sibsp']=np.log(all_data.SibSp+1)
all_data['norm_sibsp'].hist()
all_data['norm_fare']=np.log(all_data.Fare+1)
all_data['norm_fare'].hist()

all_data.Pclass=all_data.Pclass.astype(str)

all_dummies=pd.get_dummies(all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'norm_fare', 'Embarked', 'cabin_adv', 'cabin_multiple', 'numeric_ticket', 'name_title', 'train_test']])

# Split to train, test
X_train=all_dummies[all_dummies['train_test']==1].drop(['train_test'], axis=1)
X_test=all_dummies[all_dummies['train_test']==0].drop(['train_test'], axis=1)
y_train=all_data[all_data['train_test']==1].Survived

# Scaled data
scale=StandardScaler()
all_dummies_scaled=all_dummies.copy()
all_dummies_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']]=scale.fit_transform(all_dummies_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']])
X_train_scaled=all_dummies_scaled[all_dummies_scaled['train_test']==0].drop(['train_test'], axis=1)
X_test_scaled=all_dummies_scaled[all_dummies_scaled['train_test']==1].drop(['train_test'], axis=1)
y_train=all_data[all_data['train_test']==1].Survived

scale=StandardScaler()
all_dummies_scaled=all_dummies.copy()
X_train_scaled=all_dummies_scaled[all_dummies_scaled['train_test']==1].drop(['train_test'], axis=1)
X_train_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']]=scale.fit_transform(X_train_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']])
X_test_scaled=all_dummies_scaled[all_dummies_scaled['train_test']==0].drop(['train_test'], axis=1)
X_test_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']]=scale.transform(X_test_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']])
y_train=all_data[all_data['train_test']==1].Survived
y_test=all_data[all_data['train_test']==0].Survived
