import json

import numpy as np
import torch
from model import MyGRU
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

MODE = "test"

np_utterances = np.load("transformed_data/" + MODE + "/np_utterances.npy")
np_previous_acts = np.load("transformed_data/" + MODE + "/np_previous_acts.npy")
np_acts = np.load("transformed_data/" + MODE + "/np_acts.npy")
np_label = np.load("transformed_data/" + MODE + "/np_label.npy")

torch_utterances = torch.from_numpy(np_utterances).long()
torch_previous_acts = torch.from_numpy(np_previous_acts).long()
torch_acts = torch.from_numpy(np_acts).long()

torch_y = torch.from_numpy(np_label).long()
torch_y = torch_y.unsqueeze(2)

torch_previous_acts = F.one_hot(torch_previous_acts, num_classes=18)
torch_previous_acts = torch_previous_acts.squeeze(1)

dataset = TensorDataset(torch_utterances, torch_previous_acts, torch_y, torch_acts)
test_loader = DataLoader(dataset, batch_size=64)

NUM_ACTS = 18
NUM_TAGS = 47
VOCAB_SIZE = 7752

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyGRU(NUM_ACTS, NUM_TAGS, VOCAB_SIZE).to(device)
model.load_state_dict(torch.load("models/gru_6.pth"))

with open("transformed_data/dictionaries.json", "r", encoding="utf-8") as f:
    dictionaries = json.load(f)

tags_index = dictionaries['tags_idx']

class_weights = torch.tensor([1.0 if tag == "O" else 5.0 for tag in tags_index.keys()],
                             dtype=torch.float).to(device)

bio_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
act_criterion = nn.CrossEntropyLoss()

test_bio_loss = 0.0
test_act_loss = 0.0
num_batches = 0

bio_predictions = []
act_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        utterances, previous_acts, bio_tags, acts = batch

        utterances = utterances.to(device)
        previous_acts = previous_acts.to(device)
        bio_tags = bio_tags.to(device)
        acts = acts.to(device)

        bio_logits, acts_logits = model(utterances, previous_acts)

        _, bio_prediction = torch.max(bio_logits, 2)
        _, act_prediction = torch.max(acts_logits, 1)

        bio_predictions.extend(bio_prediction.cpu().numpy())
        act_predictions.append(act_prediction.cpu().numpy())

        bio_loss = bio_criterion(bio_logits.view(-1, NUM_TAGS), bio_tags.view(-1))

        act_loss = act_criterion(acts_logits, acts.view(-1))

        test_bio_loss += bio_loss.item()
        test_act_loss += act_loss.item()

average_bio_loss = test_bio_loss / len(test_loader)
average_act_loss = test_act_loss / len(test_loader)

print(f"Test BIO Loss: {average_bio_loss:.4f}, Test Act Loss: {average_act_loss:.4f}")

# print(bio_predictions)
np_act_predictions = np.hstack(act_predictions)
np_bio_predictions = np.vstack(bio_predictions)

idx_tags = dictionaries['idx_tags']

idx_acts = dictionaries['index_acts']


def decode_bio(row: np.ndarray):
    return [idx_tags[str(idx)] for idx in row]


def decode_act(idx):
    return idx_acts[str(idx)]


bio_decoded = [decode_bio(prediction) for prediction in np_bio_predictions]
act_decoded = [decode_act(prediction) for prediction in np_act_predictions]

results = {"bio_decoded": bio_decoded, "act_decoded": act_decoded}

with open("results/decoded_" + MODE + "_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f)
