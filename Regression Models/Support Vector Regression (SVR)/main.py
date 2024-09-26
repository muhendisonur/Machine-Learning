# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:58:49 2024

@author: muhen
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('time_temprature.csv')

time = df.iloc[:, 0:1].values
temprature = df.iloc[:, 1:2].values

# because of weakness the algorithm to marginal data values, we scale the values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
time_scaled = scaler.fit_transform(time)
temprature_scaled = scaler.fit_transform(temprature)

#create the SVR model
from sklearn.svm import SVR
regression = SVR(kernel='rbf', epsilon=0.2) # tuned epsilon value for better result
regression.fit(time_scaled, temprature_scaled)

# visualize the result
plt.figure()
plt.scatter(time_scaled, temprature_scaled, color='red')
plt.plot(time_scaled, regression.predict(temprature_scaled), color='blue')
plt.show()