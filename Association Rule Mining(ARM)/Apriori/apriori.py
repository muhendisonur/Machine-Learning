import pandas as pd
from apyori import apriori

df = pd.read_csv('data.csv', header = None)

item_list = []

# prepare the data as apriori object wants
for i in range(0,1001):
    item_list.append([str(df.values[i,j]) for j in range(0,7)])

# build the model
model = apriori(item_list ,min_support = 0.1, min_confidence= 0.2, min_lift=3, min_length=2, max_length=4)

# print the association result
print(list(model))