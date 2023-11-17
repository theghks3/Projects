#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv('C:/Users/82103/Desktop/프로젝트/Bike_Sharing_Demand/Bike_data/train.csv')
test=pd.read_csv('C:/Users/82103/Desktop/프로젝트/Bike_Sharing_Demand/Bike_data/test.csv')

train_test=pd.concat([train,test], axis=0)

train_test['tempdate']=pd.to_datetime(train_test['datetime'])

train_test['year']=train_test.datetime.apply(lambda x: x.split()[0].split('-')[0])
train_test['month']=train_test.datetime.apply(lambda x: x.split()[0].split('-')[1])
train_test['day']=train_test.datetime.apply(lambda x: x.split()[0].split('-')[2])
train_test['hour']=train_test.datetime.apply(lambda x: x.split()[1].split(':')[0])
train_test['weekday']=train_test['tempdate'].dt.day_name()

train_test['year']=pd.to_numeric(train_test.year, errors='coerce')
train_test['month']=pd.to_numeric(train_test.month, errors='coerce')
train_test['day']=pd.to_numeric(train_test.day, errors='coerce')
train_test['hour']=pd.to_numeric(train_test.hour, errors='coerce')

def changeseason(month):
    if month in [12, 1, 2]:
        return 4
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3
train_test['season']=train_test.month.apply(changeseason)

def cattonum(day):
    if day=='Sunday':
        return 0
    elif day=='Monday':
        return 1
    elif day=='Tuesday':
        return 2
    elif day=='Wednesday':
        return 3
    elif day=='Thursday':
        return 4
    elif day=='Friday':
        return 5
    elif day=='Saturday':
        return 6
train_test['weekday']=train_test.weekday.apply(cattonum)

windspeed_0=train_test[train_test.windspeed==0]
windspeed_not0=train_test[train_test.windspeed!=0]
windspeed_0_df=windspeed_0.drop(['datetime', 'tempdate', 'holiday', 'workingday','windspeed', 'casual', 'registered', 'count', 'weekday'], axis=1)
windspeed_not0_df=windspeed_not0.drop(['datetime', 'tempdate', 'holiday', 'workingday','windspeed', 'casual', 'registered', 'count', 'weekday'], axis=1)
windspeed_not0_series=windspeed_not0['windspeed']

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(windspeed_not0_df, windspeed_not0_series)
windspeed_0['windspeed']=rf.predict(windspeed_0_df)

train_test=pd.concat([windspeed_0, windspeed_not0], axis=0)
train_test['weekday']=train_test['weekday'].astype('category')

categorical_columns=['holiday', 'season', 'workingday', 'weather', 'weekday', 'month', 'year', 'hour']
drop_columns=['casual', 'registered', 'count', 'datetime', 'tempdate']

for col in categorical_columns:
    train_test[col]=train_test[col].astype('category')

train=train_test[pd.notnull(train_test['count'])].sort_values(by='datetime')
test=train_test[pd.notnull(train_test['count'])].sort_values(by='datetime')

datecol=test['datetime']
ycount=train['count']
ycasual=train['casual']
yregistered=train['registered']

train=train.drop(drop_columns, axis=1)
test=test.drop(drop_columns, axis=1)

def rmsle(predicted, expected, convertExp=True):
    if convertExp:
        predicted=np.exp(predicted)
        expected=np.exp(expected)
    log1=np.nan_to_num(np.array([np.log(v+1) for v in predicted]))
    log2=np.nan_to_num(np.array([np.log(v+1) for v in expected]))
    calc=(log1-log2)**2
    return np.sqrt(np.mean(calc))

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

rmsle_scorer=metrics.make_scorer(rmsle, greater_is_better=False)

# LinearRegression
lr=LinearRegression()

ycountlog=np.log1p(ycount)
lr.fit(train,ycountlog)
pred=lr.predict(train)

print('RMSLE value for Linear Regression : {}'.format(rmsle(np.exp(pred),np.exp(ycountlog), False)))

# RidgeRegression
ridge=Ridge()

ridge_params={'max_iter':[3000], 'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_ridge=GridSearchCV(ridge, ridge_params, scoring=rmsle_scorer, cv=5)

grid_ridge.fit(train, ycountlog)
pred=grid_ridge.predict(train)
print(grid_ridge.best_params_)
print('RMSLE value for Ridge Regression : {}'.format(rmsle(np.exp(pred), np.exp(ycountlog), False)))

# LassoRegression
lasso=Lasso()

lasso_params={'max_iter':[3000], 'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_lasso=GridSearchCV(lasso, lasso_params, scoring=rmsle_scorer, cv=5)

grid_lasso.fit(train, ycountlog)
pred=grid_lasso.predict(train)
print(grid_lasso.best_params_)
print('RMSLE value for Lasso Regression : {}'.format(rmsle(np.exp(pred), np.exp(ycountlog), False)))

# RandomForestRegression
rf=RandomForestRegressor()

rf_params={'n_estimators':[1, 10, 100]}
grid_rf=GridSearchCV(rf, rf_params, scoring=rmsle_scorer, cv=5)

grid_rf.fit(train, ycountlog)
pred=grid_rf.predict(train)
print(grid_rf.best_params_)
print('RMSLE value for Random Forest Regression : {}'.format(rmsle(np.exp(pred), np.exp(ycountlog), False)))

# GradientBoostingRegression
gb=GradientBoostingRegressor()

gb_params={'max_depth':range(1,11,1), 'n_estimators':[1,10,100]}
grid_gb=GridSearchCV(gb, gb_params, scoring=rmsle_scorer, cv=5)

grid_gb.fit(train, ycountlog)
pred=grid_gb.predict(train)
print(grid_gb.best_params_)
print('RMSLE value for Gradient Boosing Regression')
