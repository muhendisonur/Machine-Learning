# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:43:30 2024

@author: muhen
"""


# import the needed libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# import the dataset
veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

# Split the train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Standartizion
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# build PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test) # we used the same pca model to avoid lose relation between train and test dataset

# logistic regression classifier
model_1 = LogisticRegression(random_state=0) #we setted random state to 0 for comparing different results
model_2 = LogisticRegression(random_state=0)

# training
model_1.fit(X_train, y_train)
model_2.fit(X_train_pca, y_train)

# prediction
pred_normal = model_1.predict(X_test)
pred_pca = model_2.predict(X_test_pca)


# confusion matrices based logistic regression classifier
cm_normal = confusion_matrix(y_test, pred_normal)
accuracy_normal = accuracy_score(y_test, pred_normal)

cm_pca = confusion_matrix(y_test, pred_pca)
accuracy_pca = accuracy_score(y_test, pred_pca)


print(f"""
Confusion Matrix of Normal Dataset:
{cm_normal}
Accuracy:
{accuracy_normal*100}%
      
Confusion Matrix of Normal Dataset:
{cm_pca}
Accuracy:
{accuracy_pca*100}%

Accuracy lose between normal and pca applied prediction:
{(accuracy_normal - accuracy_pca)*100}%
""")


