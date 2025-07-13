# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 02:49:39 2024

@author: ADARSH YADAV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('train.csv')
df.head()
df.shape
df.columns
df.info()
df.drop('sl_no',axis=1,inplace=True)
df.isnull().sum()
df.drop_duplicates(inplace=True)
df.shape
df['salary'].mean()
df['salary'].median()
plt.hist(x=df['salary'], bins=40)
plt.show()
df['status'].value_counts()
df['salary'] = df['salary'].fillna(0)
df.isnull().sum()
obj = []
for col in df.columns:
    if df[col].dtype == "object":
        obj.append(col)
        print(col,df[col].unique(),len(df[col].unique()))
import warnings
warnings.filterwarnings("ignore")
sns.histplot(data=df,x='gender')
plt.title("Gender Distribution")
plt.xticks([0,1],labels=[0,1])
plt.show()
sns.pairplot(data=df, hue='gender')
plt.show()
mkt_hr_data = df[df['specialisation'] == 'Mkt&HR']
mkt_fin_data = df[df['specialisation'] == 'Mkt&Fin']
sns.boxplot(x='specialisation', y='ssc_p', data=pd.concat([mkt_hr_data, mkt_fin_data]))
plt.title('Box Plot of ssc_p for Mkt&HR and Mkt&Fin Specializations')
plt.xlabel('Specialization')
plt.ylabel('ssc_p')
plt.show()
sns.boxplot(x='specialisation', y='hsc_p', data=pd.concat([mkt_hr_data, mkt_fin_data]))
plt.title('Box Plot of hsc_p for Mkt&HR and Mkt&Fin Specializations')
plt.xlabel('Specialization')
plt.ylabel('hsc_p')
plt.show()
sns.boxplot(x='specialisation', y='degree_p', data=pd.concat([mkt_hr_data, mkt_fin_data]))
plt.title('Box Plot of degree_p for Mkt&HR and Mkt&Fin Specializations')
plt.xlabel('Specialization')
plt.ylabel('degree_p')
plt.show()
sns.boxplot(x='specialisation', y='etest_p', data=pd.concat([mkt_hr_data, mkt_fin_data]))
plt.title('Box Plot of etest_p for Mkt&HR and Mkt&Fin Specializations')
plt.xlabel('Specialization')
plt.ylabel('etest_p')
plt.show()
sns.boxplot(x='specialisation', y='mba_p', data=pd.concat([mkt_hr_data, mkt_fin_data]))
plt.title('Box Plot of mba_p for Mkt&HR and Mkt&Fin Specializations')
plt.xlabel('Specialization')
plt.ylabel('mba_p')
plt.show()
sns.histplot(data=df,x='status',hue='gender')
plt.title('Placement Status Distribution with Gender')
plt.xlabel('Placement Status')
plt.show()
sns.histplot(data=df,x='status',hue='workex')
plt.title('Placement Status Distribution with Work Experience')
plt.xlabel('Placement Status')
plt.show()
plt.pie(df['specialisation'].value_counts(), labels=df['specialisation'].value_counts().index, autopct='%1.1f%%')
plt.title("Pie chart of Specialization")
plt.show()
placed_data = df[df['status'] == 'Placed']
sns.histplot(placed_data['salary'], kde=True, bins=20)
plt.title('Salary Distribution for Placed Candidates')
plt.xlabel('Salary')
plt.ylabel('Count')
plt.show()
cols = []
for col in df.columns:
    if df[col].dtype != "object":
        cols.append(col)
corr = df[cols].corr()
sns.heatmap(corr,annot=True)
plt.title('Correlation in numeric variables')
plt.show()
for col in df.columns:
    if df[col].dtype == "object":
        print(col,df[col].unique(),len(df[col].unique()))
df['ssc_b_Central'] = df['ssc_b'].map({'Central':1,'Others':0})
df['hsc_b_Central'] = df['hsc_b'].map({'Central':1,'Others':0})
df['workex'] = df['workex'].map({'No':0,'Yes':1})
df['status'] = df['status'].map({'Placed':1,'Not Placed':0})
df['specialisation_fin'] = df['specialisation'].map({'Mkt&HR':0,'Mkt&Fin':1})
df.head()
df.drop(['ssc_b','hsc_b','specialisation'],axis=1,inplace=True)
df.head()
ohe = pd.get_dummies(df[['hsc_s','degree_t']],drop_first=True).astype(int)
ohe
df1 = pd.concat([ohe,df.drop(['hsc_s','degree_t'],axis=1)],axis=1)
df1.head()
corr1 = df1.corr()
sns.heatmap(corr1,annot=True)
plt.title('Correlation of Features with Target Variable')
plt.show()
correlated_variable = corr1['salary'].abs().sort_values(ascending=False)
correlated_variable
X = df1.drop('salary',axis=1)
X
y = df1['salary']
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
X_train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train_scaled,y_train)
reg.score(X_test_scaled,y_test)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
algos = {
    'linear_regression': {
        'model': LinearRegression(),
        'params': {

        }
    },
    'lasso': {
        'model': Lasso(max_iter=100000),
        'params': {
            'alpha': [1,2],
            'selection': ['random','cyclic']
        }
    },
    'ridge': {
        'model': Ridge(max_iter=100000),
        'params': {
            'alpha': [1,2]
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter': ['best','random']
        }
    },
    'svr': {
        'model': SVR(max_iter=10000000),
        'params': {
         
        }
    },
    'random_forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [1,5,10,20,50]
        }
    },
    'gradient_boosting': {
        'model': GradientBoostingRegressor(),
        'params': {
            'n_estimators': [1,5,10,20,50],
            'learning_rate': [0.001,0.01,0.1,0.5]
        }
    }
}
X_scaled = scaler.transform(X)
scores= []
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)
for algo_name, mp in algos.items():
   reg = GridSearchCV(mp['model'], mp['params'], cv=cv, return_train_score=False)
   reg.fit(X_scaled, y)
   scores.append(
      {
         'model': algo_name,
         'best_score': reg.best_score_,
         'best_params': reg.best_params_
      }
   ) 
score = pd.DataFrame(scores, columns=['model','best_score','best_params'])
score
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1)
ridge.fit(X_train_scaled,y_train)
ridge.score(X_test_scaled,y_test)
import pickle
with open('model.pkl','wb') as f:
    pickle.dump(ridge,f)
with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)
import pandas as pd
import pickle


with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)


df = pd.read_csv('train.csv')


df['salary'] = df['salary'].fillna(0)
df['ssc_b_Central'] = df['ssc_b'].map({'Central': 1, 'Others': 0})
df['hsc_b_Central'] = df['hsc_b'].map({'Central': 1, 'Others': 0})
df['workex'] = df['workex'].map({'No': 0, 'Yes': 1})
df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})
df['specialisation_fin'] = df['specialisation'].map({'Mkt&HR': 0, 'Mkt&Fin': 1})
df.drop(['ssc_b', 'hsc_b', 'specialisation'], axis=1, inplace=True)
ohe = pd.get_dummies(df[['hsc_s', 'degree_t']], drop_first=True).astype(int)
df1 = pd.concat([ohe, df.drop(['hsc_s', 'degree_t'], axis=1)], axis=1)


X = df1.drop(['salary', 'sl_no'], axis=1)


X_scaled = loaded_scaler.transform(X)


predictions = loaded_model.predict(X_scaled)


df['predicted_salary'] = predictions

# Save the predictions to a CSV file
df.to_csv('predictions.csv', index=False)


print(df.head())