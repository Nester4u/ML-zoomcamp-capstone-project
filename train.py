#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns


# Data Preparation, Cleaning and EDA

data = pd.read_csv("House_Price.csv")

data.head()

data.tail()

data.shape

data.info()

data.isna().sum()

data.describe()

sns.jointplot(x = "rainfall", y = "price", data = data)

sns.jointplot(x = "crime_rate", y = "price", data = data)

sns.scatterplot(x= "n_hot_rooms", y = "price", data = data)

sns.countplot(x= "airport", data = data)

sns.countplot(x="waterbody", data = data)

sns.countplot(x = "bus_ter", data = data)

data["waterbody"].unique()


# Missing Values Interpulation

data["n_hos_beds"].isna().sum()

data["n_hos_beds"].mean()

data["n_hos_beds"] = data["n_hos_beds"].fillna(data["n_hos_beds"].mean())

data["n_hos_beds"].isna().sum()

data["waterbody"].isna().sum()

data["waterbody"] = data["waterbody"].fillna("none")

sns.countplot(x ="waterbody", data = data)

data.info()

df = pd.DataFrame([6,6,6,4,4,5,5,5,5,7,7,300])

df.describe()

P99 = 7

df[df>7] = 21

df.describe()

np.percentile(data["rainfall"], 1)

lower_value = np.percentile(data["rainfall"], 1)

data[data["rainfall"] < lower_value]

data["rainfall"][data["rainfall"] < lower_value] = 0.3 * lower_value

data.describe()

np.percentile(data.n_hot_rooms, 99)

uv = np.percentile(data.n_hot_rooms, 99)

data[data["n_hot_rooms"] > uv]

data["n_hot_rooms"][data["n_hot_rooms"] > 3*uv]  = 3*uv

data.describe()

sns.scatterplot(x = data["rainfall"], y = data["price"])

data.bus_ter.unique()

data.drop("bus_ter", axis =1, inplace = True)

data.head()

data.shape

data.info()

sns.scatterplot(x = data["crime_rate"], y = data["price"])

data.crime_rate.describe()

sns.displot(x = "crime_rate", data = data)

sns.distplot(x =data.price,  hist=True, kde=True)

data["crime_rate"].describe()

np.log(data["crime_rate"]).describe()

np.sqrt(data["crime_rate"]).describe()

np.log(data["crime_rate"] + 1).describe()

np.exp(data["crime_rate"]).describe()

data["crime_rate"] = np.log(data["crime_rate"] + 1)

data["crime_rate"].describe()

sns.scatterplot(x = data["crime_rate"], y = data["price"])

data.head()

df =pd.get_dummies(data["airport"])

df.head()

data["airport"] = df["YES"]

data.head()

data["waterbody"].unique()

['None','Lake', 'River', 'Lake and River ']


from sklearn.preprocessing import OrdinalEncoder

order_mapping= ['None','Lake', 'River', 'Lake and River'][::-1]

encoder = OrdinalEncoder(categories=[order_mapping])

data["waterbody_Encoded"] = encoder.fit_transform(data[["waterbody"]])


data.head()

del data["waterbody"]

data.head()


data.info()

data.corr()

import matplotlib.pyplot as plt
plt.figure(figsize = (13, 13))

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', center=0, square=True)

data[["dist1", "dist2","dist3", "dist4"]].describe()

data["avg_dist"] = (data["dist1"]+data["dist2"]+data["dist3"]+data["dist4"])/4


data.head()

data.drop(["dist1", "dist2","dist3", "dist4"], axis=1,inplace=True)

data.head()

import matplotlib.pyplot as plt
plt.figure(figsize = (13, 13))

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', center=0, square=True)


data[["parks", "air_qual"]].describe()

del data["parks"]

data.head()


import matplotlib.pyplot as plt
plt.figure(figsize = (13, 13))

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', center=0, square=True)


sns.regplot(x = data["poor_prop"], y = data["price"])

# Linear Regression
y = data["price"]
x = data.drop("price",axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x, y)


model.intercept_, model.coef_

data.head()

y_pred = model.predict(x_train)
y_pred_sel = model.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_train, y_pred)


r2_score(y_test, y_pred_sel)


# Random Forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=3, random_state=42)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)

y_pred_sel = rf.predict(x_test)

r2_score(y_train, y_pred)


r2_score(y_test, y_pred_sel)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets import make_regression

reg = DecisionTreeRegressor(criterion = 'mse', max_depth = 3, random_state = 42)

reg.fit(x_train, y_train)

reg.score(x_train, y_train)

reg.score(x_test, y_test)


# Evaluation of the Model Selection Process

# - The linear regression, random forest and decision tree processes were carried out on the model and the decision tree was found to have the best performance

# Parameter Tuning and Training the Final Model 

data.head()

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

param_grid = {
    'criterion': ['mse', 'friedman_mse', 'mae'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=2, verbose=2, n_jobs=-1)

grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
best_reg_model = grid_search.best_estimator_

# Evaluating the best model on the train data
accuracy = best_reg_model.score(x_train, y_train)
print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)

# Evaluating the best model on the test data
accuracy = best_reg_model.score(x_test, y_test)
print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)


# Saving the Model

import pickle

filename = 'house_price.pkl'
pickle.dump(best_reg_model,open(filename,'wb'))

