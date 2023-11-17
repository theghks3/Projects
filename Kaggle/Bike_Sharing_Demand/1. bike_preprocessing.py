#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv('C:/Users/82103/Desktop/프로젝트/Bike_Sharing_Demand/Bike_data/train.csv')
test=pd.read_csv('C:/Users/82103/Desktop/프로젝트/Bike_Sharing_Demand/Bike_data/test.csv')

train['tempdate']=pd.to_datetime(train['datetime'])

train['year']=train.datetime.apply(lambda x: x.split()[0].split('-')[0])
train['month']=train.datetime.apply(lambda x: x.split()[0].split('-')[1])
train['day']=train.datetime.apply(lambda x: x.split()[0].split('-')[2])
train['hour']=train.datetime.apply(lambda x: x.split()[1].split(':')[0])
train['weekday']=train['tempdate'].dt.day_name()
train=train.drop(['datetime', 'tempdate'], axis=1)

# Changing object data to numeric data
train['year']=pd.to_numeric(train.year, errors='coerce')
train['month']=pd.to_numeric(train.month, errors='coerce')
train['day']=pd.to_numeric(train.day, errors='coerce')
train['hour']=pd.to_numeric(train.hour, errors='coerce')

# Relationship between year-count, month-count, day-count, hour-count
fig=plt.figure(figsize=(12, 8))
ax1=fig.add_subplot(2,2,1)
ax1=sns.barplot(x='year', y='count', data=train.groupby('year')['count'].mean().reset_index())
ax2=fig.add_subplot(2,2,2)
ax2=sns.barplot(x='month', y='count', data=train.groupby('month')['count'].mean().reset_index())
ax3=fig.add_subplot(2,2,3)
ax3=sns.barplot(x='day', y='count', data=train.groupby('day')['count'].mean().reset_index())
ax4=fig.add_subplot(2,2,4)
ax4=sns.barplot(x='hour', y='count', data=train.groupby('hour')['count'].mean().reset_index())

# Changing seasons
def badtoright(month):
    if month in [12, 1, 2]:
        return 4
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3

train['season']=train.month.apply(badtoright)

# Changing weekday to numeric data
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

train['weekday']=train.weekday.apply(cattonum)


# Relationship between season-count, holiday-count, workingday-count, weather-count
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(2,2,1)
ax1=sns.barplot(x='season', y='count', data=train.groupby('season')['count'].mean().reset_index())
ax2=fig.add_subplot(2,2,2)
ax2=sns.barplot(x='holiday', y='count', data=train.groupby('holiday')['count'].mean().reset_index())
ax3=fig.add_subplot(2,2,3)
ax3=sns.barplot(x='workingday', y='count', data=train.groupby('workingday')['count'].mean().reset_index())
ax4=fig.add_subplot(2,2,4)
ax4=sns.barplot(x='weather', y='count', data=train.groupby('weather')['count'].mean().reset_index())

# Relationship between temp-count, atemp-count, humidity-count, windspeed-count
fig=plt.figure(figsize=(12,10))
ax1=fig.add_subplot(2,2,1)
ax1=sns.displot(train.temp, bins=range(train.temp.min().astype('int'),train.temp.max().astype('int')+1))
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.displot(train.atemp,bins=range(train.atemp.min().astype('int'),train.atemp.max().astype('int')+1))
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.displot(train.humidity,bins=range(train.humidity.min().astype('int'),train.humidity.max().astype('int')+1))
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.displot(train.windspeed,bins=range(train.windspeed.min().astype('int'),train.windspeed.max().astype('int')+1))

# Heatmap of data
train=train.drop(['datetime', 'tempdate'], axis=1) # For heatmap
fig=plt.figure(figsize=(20,20))
ax=sns.heatmap(data=train.corr(), annot=True, square=True)

# Pointplot of counts based on the heatmap
# The reason for weather-month is there is an outlier where train.weather==4
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(2,2,1)
ax1=sns.pointplot(x='hour', y='count', hue='season', data=train.groupby(['season', 'hour'])['count'].mean().reset_index())
ax2=fig.add_subplot(2,2,2)
ax2=sns.pointplot(x='hour', y='count', hue='holiday', data=train.groupby(['holiday', 'hour'])['count'].mean().reset_index())
ax3=fig.add_subplot(2,2,3)
ax3=sns.pointplot(x='hour', y='count', hue='weekday', hue_order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], data=train.groupby(['weekday', 'hour'])['count'].mean().reset_index())
ax4=fig.add_subplot(2,2,4)
ax4=sns.pointplot(x='month', y='count', hue='weather', data=train.groupby(['weather', 'month'])['count'].mean().reset_index())

# Too much 0 at windspeed: 실제 0인지, 값을 제대로 측정하지 못해서 0인지 - 후자로 생각
train['weekday']=train.weekday.astype('category')
train.weekday.cat.categories=['5', '1', '6', '0', '4', '2', '3']
windspeed_0=train[train.windspeed==0]
windspeed_not0=train[train.windspeed!=0]
windspeed_0_df=windspeed_0.drop(['windspeed', 'holiday', 'workingday', 'casual', 'registered', 'count', 'weekday'], axis=1)
windspeed_not0_df=windspeed_not0.drop(['windspeed', 'holiday', 'workingday', 'casual', 'registered', 'count', 'weekday'], axis=1)
windspeed_not0_series=windspeed_not0['windspeed']

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(windspeed_not0_df, windspeed_not0_series)
windspeed_0['windspeed']=rf.predict(windspeed_0_df)

train=pd.concat([windspeed_0, windspeed_not0], axis=0)

# Despite our expectations, the correlation between windspeed and count grew from 0.1 only to 0.11
# So, it is plausible to assume that we don't have to care much about windspeed
fig=plt.figure(figsize=(20,20))
ax=sns.heatmap(train.corr(), annot=True, square=True)
