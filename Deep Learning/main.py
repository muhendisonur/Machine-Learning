import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer

# preprocessing
df = pd.read_csv("https://bilkav.com/Churn_Modelling.csv")
exited = df.iloc[:, -1:].values
df_new = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)


ct = make_column_transformer(
    (OneHotEncoder(sparse_output=False), ['Geography']),
    (OrdinalEncoder(), ['Gender']),
    remainder="passthrough"
)

ct.set_output(transform="pandas")

# after the transformer, index ordering doesnt match with exited dataframe, to fix this:
df_encoded= ct.fit_transform(df_new)
df_encoded.set_index(df_new.index)

# scaling
sc = StandardScaler()
df_encoded_scaled = sc.fit_transform(df_encoded)


# split the train and test values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_encoded_scaled, exited, test_size=0.2, random_state=61)


# build the ANN
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=12)) #first layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu")) #second layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid")) #output layer

# compile the ANN and trainnig it
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
classifier.fit(x_train, y_train, epochs=100)

# prediction and confusion matrix
pred = classifier.predict(x_test)
pred = (pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
print(cm)