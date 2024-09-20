# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:00:17 2024

@author: muhen
"""

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('maaslar.csv')

education_level = df.iloc[:, 1:2].values
salary = df.iloc[:, 2:3].values

# handling as linear regression to make comparasion at forward START
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(education_level, salary)

plt.figure()
plt.scatter(education_level, salary, color='red')
plt.plot(education_level, regression.predict(education_level), color='blue')
plt.show()
# handling as linear regression to make comparasion at forward END

# polynomial regression START
from sklearn.preprocessing import PolynomialFeatures
poly_converter = PolynomialFeatures(degree = 4) #makes a transform the array into decided degree polynomial array
education_level_poly = poly_converter.fit_transform(education_level)
salary_poly = poly_converter.fit_transform(salary)
regression.fit(education_level_poly, salary)

plt.figure()
plt.scatter(education_level, salary, color='red')
plt.plot(education_level, regression.predict(education_level_poly), color='green')
plt.show()
# polynomial regression END
