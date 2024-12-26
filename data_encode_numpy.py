import numpy as np
import pandas as pd
import ast
import json

DATA_PATH = "transformed_data/train/train.csv"

data = pd.read_csv(DATA_PATH, index_col=0)
user_utterances = data[data['speaker'] == "USER"]
all_acts = user_utterances['dialogue_act'].unique()
all_intents = user_utterances['active_intent'].unique()

user_utterances['utterance'] = user_utterances['utterance'].apply(lambda x: str(x).lower() + " <eos>")
user_utterances['label'] = user_utterances.apply(lambda row: ast.literal_eval(row['bio_tag']) + [row['active_intent']], axis=1)

for i in range(len(user_utterances) * 2):
    if i % 2 != 0:
        continue
    current = user_utterances.loc[i]

    if current['turn'] == 0:
        previous_act = "general-greet"
    else:
        previous_act = user_utterances.loc[i - 2, 'dialogue_act']
    user_utterances.loc[i, 'previous_act'] = previous_act

all_tags = set()
user_utterances.apply(lambda row: all_tags.update(ast.literal_eval(row['bio_tag'])), axis=1)
all_tags = list(all_tags)
all_tags.extend(all_intents)
all_tags.insert(0, "<pad>")
print(len(all_tags))

vocabs = set()
user_utterances.apply(lambda row: vocabs.update(row['utterance'].split()), axis=1)
vocabs = list(vocabs)
vocabs.insert(0, "<pad>")
vocabs.insert(1, "<unk>")
print(len(vocabs))

acts_index = {act: index for index, act in enumerate(all_acts)}
index_acts = {index: act for index, act in enumerate(all_acts)}
tags_idx = {tag: idx for idx, tag in enumerate(all_tags)}
idx_tags = {idx: tag for idx, tag in enumerate(all_tags)}
tokens_idx = {token: idx for idx, token in enumerate(vocabs)}
idx_tokens = {idx: token for idx, token in enumerate(vocabs)}

utterances_idx = [[tokens_idx[token] for token in sentence.split()] for sentence in user_utterances['utterance']]
max_length = max(len(sentence) for sentence in utterances_idx)
np_utterances = np.array(
    [sentence + [0] * (max_length - len(sentence)) for sentence in utterances_idx],
    dtype=np.uint32
)
bio_tag_idx = [[tags_idx[tag] for tag in tags] for tags in user_utterances['label']]
max_length = max(len(sentence) for sentence in bio_tag_idx)
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


np.save("transformed_data/train/np_utterances.npy", np_utterances)
np.save("transformed_data/train/np_previous_acts.npy", np_previous_acts)
np.save("transformed_data/train/np_acts.npy", np_acts)
np.save("transformed_data/train/np_label.npy", np_label)

# Combine all dictionaries into one
all_dicts = {
    "acts_index": acts_index,
    "index_acts": index_acts,
    "tags_idx": tags_idx,
    "idx_tags": idx_tags,
    "tokens_idx": tokens_idx,
    "idx_tokens": idx_tokens
}

# Save to JSON
with open("transformed_data/dictionaries.json", "w") as file:
    json.dump(all_dicts, file, indent=4)

