import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm


df = pd.read_csv('salary.csv')
df = df.drop(['Calisan ID', 'unvan'], axis=1) #removed categorial columns

original_inputs = df.iloc[:, 0:3].values
original_output = df.iloc[:, -1:].values

#splitted test and train
x_train, x_test, y_train, y_test = train_test_split(original_inputs, original_output, test_size = 0.33, random_state = 0)

# multiple linear regression START
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
multiple_regression_result = linear_regression.predict(x_test)

# visualization of prediction result
plt.figure()
plt.scatter(range(10), y_test, color='red')
plt.plot(range(10), multiple_regression_result, color='blue')
plt.title("Multiple Linear Regression Prediction Result")
plt.ylim(top = 35000, bottom = 0) # to make scatters adjusted on both figure, I scaled y axis min, max values
plt.show
# multiple linear regression END

# random forest regression START
random_forest_regression = RandomForestRegressor()
random_forest_regression.fit(x_train, y_train)
random_forest_result = random_forest_regression.predict(x_test)

# visualization of prediction result
plt.figure()
plt.scatter(range(10), y_test, color='red')
plt.plot(range(10), random_forest_result, color='blue')
plt.title("Random Forest Regression Prediction Result")
plt.ylim(top = 35000, bottom = 0) # to make scatters adjusted on both figure, I scaled y axis min, max values
plt.show()
# random forest regression END


print("""
FIRST RESULT OF MODELS
----------------------------------------------------------------------------------------------      
""")

# First Conclusion
multiple_linear_regression_first_analyz = sm.OLS(multiple_regression_result, x_test).fit()
print("""
Multiple Linear Regression Analyz
----------------------------------------------------------------------------------------------
""")
print(multiple_linear_regression_first_analyz.summary())

random_forest_first_analyz = sm.OLS(random_forest_result, x_test).fit()
print("""
Random Forest Regression Analyz
----------------------------------------------------------------------------------------------
""")
print(random_forest_first_analyz.summary())


# adjusting on some parameters (backward elimination)

# multiple linear regression adjusting START
mlr_adjusted_x_train = x_train[:, 0:1]
mlr_adjusted_x_test = x_test[:, 0:1]
linear_regression.fit(mlr_adjusted_x_train, y_train)
adjusted_multiple_regression_result = linear_regression.predict(mlr_adjusted_x_test) #actually its just linear regression(cause there is left one input paramater after the adjusting) anymore but the base of model is named as MLR so I keep it like that

# visualization of prediction result
plt.figure()
plt.scatter(range(10), y_test, color='red')
plt.plot(range(10), adjusted_multiple_regression_result, color='blue')
plt.title("Adjusted Multiple Linear Regression Prediction Result")
plt.ylim(top = 35000, bottom = 0) # to make scatters adjusted on both figure, I scaled y axis min, max values
plt.show
# multiple linear regression adjusting END


# random forest regression adjusting START
rfr_adjusted_x_train = x_train[:, 0:1]
rfr_adjusted_x_test = x_test[:, 0:1]
random_forest_regression.fit(rfr_adjusted_x_train, y_train)
adjusted_random_forest_result = random_forest_regression.predict(rfr_adjusted_x_test)

# visualization of prediction result
plt.figure()
plt.scatter(range(10), y_test, color='red')
plt.plot(range(10), adjusted_random_forest_result, color='blue')
plt.title("Adjusted Random Forest Regression Prediction Result")
plt.ylim(top = 35000, bottom = 0) # to make scatters adjusted on both figure, I scaled y axis min, max values
plt.show
# random forest regression adjusting END



print("""
      
      
      
LAST RESULT OF MODELS
----------------------------------------------------------------------------------------------      
""")
# Second Conclusion
multiple_linear_regression_last_analyz = sm.OLS(adjusted_multiple_regression_result, mlr_adjusted_x_test).fit()
print("""
Multiple Linear Regression Analyz
----------------------------------------------------------------------------------------------
""")
print(multiple_linear_regression_last_analyz.summary())

random_forest_last_analyz = sm.OLS(adjusted_random_forest_result, rfr_adjusted_x_test).fit()
print("""
Random Forest Regression Analyz
----------------------------------------------------------------------------------------------
""")
print(random_forest_last_analyz.summary())