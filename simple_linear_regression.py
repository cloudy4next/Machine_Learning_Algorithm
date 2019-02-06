#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 04:25:55 2019

@author: cloudy
"""


import matplotlib.pyplot as plt
import pandas as pd

#import data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction
y_pred = regressor.predict(X_test)

#Visualising the training set

plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train,regressor.predict(X_train), color = 'black')
plt.title('Salary vs Expreience (Trainig set)')
plt.xlabel('Years of Experience ')
plt.ylabel('Salary')
plt.show()


# Visualising the Test

plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
