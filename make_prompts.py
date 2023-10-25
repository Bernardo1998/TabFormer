import torch
import pandas as pd
import random
from random import sample 
import json

real = pd.read_csv("data/dvlog/preprocessed/acoustic.encoded.csv")
real_ids = list(set(real['User']))
random.seed(42)
selected_ids = sample(real_ids,int(len(real_ids) * 0.2))
sampled_sequences = real[real['User'].isin(selected_ids)].sample(1000)

result_dict = {}

# Open and read the .nb file
with open('checkpoints/vocab.nb', 'r') as file:
    for row_index, line in enumerate(file):
        # Remove newline character at the end of the line
        line = line.strip()
        # Split the line into columnName and columnValue
        column_name, column_value = line.split('_')
        # If the columnName is not already a key in the dictionary, add it
        if column_name not in result_dict:
            result_dict[column_name] = {}
        # Add columnValue under columnName and set its value to row_index
        if column_name != 'SPECIAL':
            column_value = float(column_value)
        result_dict[column_name][column_value] = row_index

print(result_dict)

prompt_list = []
for i in range(sampled_sequences.shape[0]):
    row = sampled_sequences.iloc[i, :]
    prompt_this_row = []
    for c in sampled_sequences.columns:
        prompt_this_row.append(result_dict[c][row[c]])
    prompt_list.append(prompt_this_row)

with open("data/dvlog/prompt.json", 'w') as F:
    F.write(json.dumps(prompt_list))