import numpy as np
import pandas as pd
import ast
import json
import os

MODE = "test"  # or "dev"

DATA_PATH = os.path.join("transformed_data", MODE, MODE + ".csv")

data = pd.read_csv(DATA_PATH, index_col=0)
user_utterances = data[data['speaker'] == "USER"]
all_acts = user_utterances['dialogue_act'].unique()
all_intents = user_utterances['active_intent'].unique()

user_utterances['utterance'] = user_utterances['utterance'].apply(lambda x: str(x).lower() + " <eos>")
user_utterances['label'] = user_utterances.apply(lambda row: ast.literal_eval(row['bio_tag']) + [row['active_intent']],
                                                 axis=1)

for i in range(len(user_utterances) * 2):
    if i % 2 != 0:
        continue
    current = user_utterances.loc[i]

    if current['turn'] == 0:
        previous_act = "general-greet"
    else:
        previous_act = user_utterances.loc[i - 2, 'dialogue_act']
    user_utterances.loc[i, 'previous_act'] = previous_act

with open("transformed_data/dictionaries.json", "r", encoding="utf-8") as f:
    dictionaries = json.load(f)

tokens_idx = dictionaries['tokens_idx']
tags_idx = dictionaries['tags_idx']
acts_index = dictionaries['acts_index']

max_length = 39
utterances_idx = [[tokens_idx[token] if token in tokens_idx else tokens_idx["<unk>"]
                   for token in sentence.split()] for sentence in user_utterances['utterance']]
np_utterances = np.array(
    [sentence + [0] * (max_length - len(sentence)) for sentence in utterances_idx],
    dtype=np.uint32
)
bio_tag_idx = [[tags_idx[tag] for tag in tags] for tags in user_utterances['label']]
np_label = np.array(
    [tags + [0] * (max_length - len(tags)) for tags in bio_tag_idx],
    dtype=np.uint32
)

act_idx = [acts_index[act] for act in user_utterances['dialogue_act']]
np_acts = np.array(act_idx, dtype=np.uint32)
np_acts = np_acts.reshape(-1, 1)

previous_act_idx = [acts_index[act] for act in user_utterances['previous_act']]
np_previous_acts = np.array(previous_act_idx, dtype=np.uint32)
np_previous_acts = np_previous_acts.reshape(-1, 1)

np.save(os.path.join("transformed_data", MODE, "np_utterances.npy"), np_utterances)
np.save(os.path.join("transformed_data", MODE, "np_previous_acts.npy"), np_previous_acts)
np.save(os.path.join("transformed_data", MODE, "np_acts.npy"), np_acts)
np.save(os.path.join("transformed_data", MODE, "np_label.npy"), np_label)
