import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('conditions.csv')

#handling some rows for encoding(Categorical to numeric)
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

outlook = df.iloc[:, 0:1].values
outlook[:, 0] = le.fit_transform(outlook[:, 0]) #label encoding
outlook = ohe.fit_transform(outlook).toarray()
outlook = pd.DataFrame(data=outlook, index=range(14), columns=['overcast', 'rainy', 'sunny']) #converted to dataframe to concat with other dataframes

windy = df.iloc[:, 3:4].values
windy = np.array(windy, dtype=float)
windy = pd.DataFrame(data=windy, index=range(14), columns=['windy']) #converted to dataframe to concat with other dataframes

play = df.iloc[:, -1:].values
play[:, -1] = le.fit_transform(play[:, -1])
play = pd.DataFrame(data=play, index=range(14), columns=['play']) #converted to dataframe to concat with other dataframes

df = df.drop(['outlook', 'windy', 'play'], axis=1) #removed categorical rows to update them with numeric version

df1 = pd.concat([outlook, df], axis=1)  #numeric encoded outlook row added
df2 = pd.concat([df1, windy], axis=1)   #numeric encoded windy row added
df3 = pd.concat([df2, play], axis=1)     #numeric encoded play row added

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df2, play, test_size=0.33, random_state=1)

#building multiple linear regression model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)
result = regression.predict(x_test)

print(f"""
True Result:
{y_test.values}

Predicted Result: 
{result}
""")


#backward elimination
import statsmodels.api as sm
#by multiple linear regression formula we need a constant value as 1
X_l = df2.iloc[:, [0,1,2,3,4,5]].values
be_model = sm.OLS(np.array(play, dtype=float), X_l).fit()
print(be_model.summary()) #by the summary, fourth row is over the 0.5

X_l = df2.iloc[:, [0,1,2,4,5]].values #removed the fourth row
be_model = sm.OLS(np.array(play, dtype=float), X_l).fit()
print(be_model.summary()) #summary of after the elimination

#reapplied multiple linear regression by latest updates(elimination etc.)
X_train, X_test, Y_train, Y_test = train_test_split(X_l, play, test_size=0.33, random_state=1)
regression.fit(X_train, Y_train)
final_result = regression.predict(X_test)