#!/usr/bin/env python
# coding: utf-8

from machine_learning import run_model
import os
import pandas as pd
from itertools import combinations, chain
categories = ['hiphop', 'pop', 'country', 'rnb', 'latin', 'rock', 'edm_dance',
              'indie_alt', 'classical', 'jazz', 'soul', 'kpop', 'reggae', 'blues']
# print(run_model(["blues", "jazz", "soul"]))


df = pd.DataFrame(columns=["genres", "n_genres", "loss", "accuracy"])


for i in range(2, len(categories)+1):
    combos = list(combinations(categories, i))
    for combo in combos:
        print(combo)
        loss, accuracy = run_model(combo)
        df = df.append({'genres': ', '.join(combo), 'n_genres': i,
                        'loss': loss, 'accuracy': accuracy}, ignore_index=True)

#     run_model(combo)
df.to_csv("full_set_analysis.csv")

# Create Categorical Data
for index, row in df.iterrows():
    for cat in categories:
        if cat in row["genres"].split(', '):
            df[cat][index] = 1
    else:
        df[cat][index] = 0
df.to_csv("FINAL_full_set_analysis.csv")