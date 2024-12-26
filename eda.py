import pandas as pd

data = pd.read_csv('transformed_data/train/train.csv', index_col=0)

data = data[data['speaker'] == "USER"]

print(data['dialogue_act'].unique())