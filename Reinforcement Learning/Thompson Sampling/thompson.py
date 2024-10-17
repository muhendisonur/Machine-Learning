import pandas as pd
from random import Random
import math
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv').values

df_first_dim = df.shape[0] # for this application its 10000
df_second_dim = df.shape[1] # for this application its 10

# Calculating total score of ones to compare at forward
total_score = 0
for i in range (0,10000):
    for j in range (0,10):
        total_score += df[i,j]   

# calculating the spread of ones values on df
ones_counter = [0] * df_second_dim
for i in range(df_first_dim):
    for j in range(df_second_dim):
        ones_counter[j] += df[i,j]

# Random selection algorithm with score
rand = Random()
random_score = 0
selected_indexes = []
for i in range (0,10000):
    rand_selected_index = rand.randrange(10)
    selected_indexes.append(rand_selected_index)
    random_score += df[i, rand_selected_index]

# Upper Confidence Bound (UCB) algorithm with score
selected_ads = []
ones = [0] * df_second_dim
zeros = [0] * df_second_dim
total_score_th = 0
selected_ads = []

for i in range(df_first_dim):
    selected_ad = 0
    max_th = 0
    for j in range(df_second_dim):
        rasbeta = Random().betavariate(ones[j] + 1, zeros[j] + 1) # random probability taken from beta distribution
        if rasbeta > max_th:
            max_th = rasbeta
            selected_ad = j
    
    total_score_th += df[i,selected_ad]
    selected_ads.append(selected_ad)
            
    if(df[i, selected_ad] == 1):
        ones[selected_ad] += 1
    else:
        zeros[selected_ad] += 1
        
    
    
print(f"""
total of the ones:
{total_score}

the spread of ones:
{ones_counter}

result of random selection algorithm score:
{random_score}

result of Thompson Sampling algorithm score:
{total_score_th}    
""")
    
# visulazition of UCB algorithm's selected ads
plt.figure()
plt.hist(selected_ads)
plt.show()