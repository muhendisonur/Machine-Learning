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
clicks = [0] * df_second_dim
clicks_total_score = [0] * df_second_dim
total_score_ucb = 0
selected_ads = []

for i in range(df_first_dim):
    selected_ad = 0
    max_ucb = 0
    for j in range(df_second_dim):
        if(clicks[j] > 0):
            avarage_score = clicks_total_score[j] / clicks[j]
            delta = math.sqrt(3/2 * math.log(i)/clicks[j])
            ucb = avarage_score + delta
        else:
            ucb = df_first_dim*10
        
        if max_ucb < ucb:
            max_ucb = ucb
            selected_ad = j
    
    clicks[selected_ad] += 1
    clicks_total_score[selected_ad] += df[i, selected_ad]
    total_score_ucb += df[i, selected_ad]
    selected_ads.append(selected_ad)
    
    
print(f"""
total of the ones:
{total_score}

the spread of ones:
{ones_counter}

result of random selection algorithm score:
{random_score}

result of UCB algorithm score:
{total_score_ucb}    
""")
    
# visulazition of UCB algorithm's selected ads
plt.figure()
plt.hist(selected_ads)
plt.show()