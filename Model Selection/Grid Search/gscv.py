# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:58:49 2024

@author: muhen
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

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
regression = SVR() # tuned epsilon value for better result
regression.fit(time_scaled, temprature_scaled.ravel())


# grid search
gscv_p = [
    {'kernel':['linear', 'poly', 'rbf'], 'degree':[1,2,3,4,5,6,7,8,9], 'epsilon':[0.1, 0.2, 0.3]}]

gscv = GridSearchCV(
    estimator = regression,
    param_grid=gscv_p,
    cv = 4,
    n_jobs=-1)

gscv.fit(time_scaled, temprature_scaled.ravel())

print(f"""
GridSearchCv Best Params:
{gscv.best_params_}
""")
