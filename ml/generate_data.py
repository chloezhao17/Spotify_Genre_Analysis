from machine_learning import run_model
import os
import pandas as pd
from itertools import combinations, chain
categories = ['hiphop', 'pop', 'country', 'rnb', 'latin', 'rock', 'edm_dance',
              'indie_alt', 'classical', 'jazz', 'soul', 'kpop', 'reggae', 'blues']

df = pd.DataFrame(columns=["genres", "n_genres", "loss", "accuracy"])

for i in range(2, len(categories)+1):
    combos = list(combinations(categories, i))
    for combo in combos:
        print(combo)
        loss, accuracy = run_model(combo)
        df = df.append({'genres': ', '.join(combo), 'n_genres': 3,
                        'loss': loss, 'accuracy': accuracy}, ignore_index=True)

#     run_model(combo)
df.to_csv("full_set_analysis.csv")
