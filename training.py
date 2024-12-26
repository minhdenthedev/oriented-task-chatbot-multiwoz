import json

import numpy as np
import torch
from model import MyGRU
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

np_utterances = np.load("transformed_data/train/np_utterances.npy")
np_previous_acts = np.load("transformed_data/train/np_previous_acts.npy")
np_acts = np.load("transformed_data/train/np_acts.npy")
np_label = np.load("transformed_data/train/np_label.npy")

torch_utterances = torch.from_numpy(np_utterances).long()
torch_previous_acts = torch.from_numpy(np_previous_acts).long()
torch_acts = torch.from_numpy(np_acts).long()

torch_y = torch.from_numpy(np_label).long()
torch_y = torch_y.unsqueeze(2)

torch_previous_acts = F.one_hot(torch_previous_acts, num_classes=18)
torch_previous_acts = torch_previous_acts.squeeze(1)

dataset = TensorDataset(torch_utterances, torch_previous_acts, torch_y, torch_acts)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

np_dev_utterances = np.load("transformed_data/dev/np_utterances.npy")
np_dev_previous_act = np.load("transformed_data/dev/np_previous_acts.npy")
np_dev_acts = np.load("transformed_data/dev/np_acts.npy")
np_dev_label = np.load("transformed_data/dev/np_label.npy")

torch_dev_utterances = torch.from_numpy(np_dev_utterances).long()
torch_dev_previous_acts = torch.from_numpy(np_dev_previous_act).long()
torch_dev_acts = torch.from_numpy(np_dev_acts).long()
torch_dev_y = torch.from_numpy(np_dev_label).long()
torch_dev_y = torch_dev_y.unsqueeze(2)
torch_dev_previous_acts = F.one_hot(torch_dev_previous_acts, num_classes=18)
torch_dev_previous_acts = torch_dev_previous_acts.squeeze(1)

dataset = TensorDataset(torch_dev_utterances, torch_dev_previous_acts, torch_dev_y, torch_dev_acts)
dev_loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_EPOCHS = 40

NUM_ACTS = 18
NUM_TAGS = 47
VOCAB_SIZE = 7752
PATIENT = 4

model = MyGRU(NUM_ACTS, NUM_TAGS, VOCAB_SIZE).to(device)

with open("transformed_data/dictionaries.json", "r", encoding="utf-8") as f:
    dictionaries = json.load(f)

tags_index = dictionaries['tags_idx']

class_weights = torch.tensor([1.0 if tag == "O" else 5.0 for tag in tags_index.keys()],
                             dtype=torch.float).to(device)
bio_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
act_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

last_dev_bio_loss = 1000.0
last_dev_act_loss = 1000.0

early_stop_count_bio = 0
early_stop_count_act = 0

results = {
    'train': {
        'bio': [],
        'act': []
    },
    'dev': {
        'bio': [],
        'act': []
    }
}

for epoch in range(N_EPOCHS):
    model.train()
    epoch_bio_loss = 0.0
    epoch_act_loss = 0.0

    for batch in tqdm(train_loader):
        utterances, previous_acts, bio_tags, acts = batch
        utterances = utterances.to(device)
        previous_acts = previous_acts.to(device)
        bio_tags = bio_tags.to(device)
        acts = acts.to(device)

        bio_logits, acts_logits = model(utterances, previous_acts)

        bio_loss = bio_criterion(bio_logits.view(-1, NUM_TAGS), bio_tags.view(-1))

        act_loss = act_criterion(acts_logits, acts.view(-1))

        total_loss = bio_loss + act_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_bio_loss += bio_loss.item()
        epoch_act_loss += act_loss.item()

    results['train']['bio'].append(float(epoch_bio_loss/len(train_loader)))
    results['train']['act'].append(float(epoch_act_loss/len(train_loader)))

    model.eval()
    epoch_bio_dev_loss = 0.0
    epoch_act_dev_loss = 0.0
    for batch in tqdm(dev_loader):
        utterances, previous_acts, bio_tags, acts = batch
        utterances = utterances.to(device)
        previous_acts = previous_acts.to(device)
        bio_tags = bio_tags.to(device)
        acts = acts.to(device)

        bio_logits, acts_logits = model(utterances, previous_acts)

        bio_dev_loss = bio_criterion(bio_logits.view(-1, NUM_TAGS), bio_tags.view(-1))
        act_dev_loss = act_criterion(acts_logits, acts.view(-1))
        epoch_bio_dev_loss += bio_dev_loss.item()
        epoch_act_dev_loss += act_dev_loss.item()
    results['dev']['bio'].append(float(epoch_bio_dev_loss / len(dev_loader)))
    results['dev']['act'].append(float(epoch_act_dev_loss / len(dev_loader)))

    if epoch_bio_dev_loss < last_dev_bio_loss:
        early_stop_count_bio = 0
    else:
        early_stop_count_bio += 1

    if epoch_act_dev_loss < last_dev_act_loss:
        early_stop_count_act = 0
    else:
        early_stop_count_act += 1

    last_dev_bio_loss = epoch_bio_dev_loss
    last_dev_act_loss = epoch_act_dev_loss

    # Check for early stopping based on bio and act losses
    if early_stop_count_bio >= PATIENT:
        print(f"BIO loss has not improved for {early_stop_count_bio} epochs. Freezing GRU layers.")
        # Freezing the GRU layers but still train the dense layer here
        for param in model.grus.parameters():
            param.requires_grad = False  # Freeze GRU layers

    if early_stop_count_bio >= PATIENT or early_stop_count_act >= PATIENT:
        print("Early stopping triggered.")
        break

    print(f"Epoch {epoch + 1}/{N_EPOCHS}: BIO Train Loss: {epoch_bio_loss/len(train_loader):.4f}, "
          f"Act Train Loss: {epoch_act_loss/len(train_loader):.4f}, "
          f"BIO Dev Loss: {epoch_bio_dev_loss/len(dev_loader):.4f}, "
          f"Act Dev Loss: {epoch_act_dev_loss/len(dev_loader):.4f}\n")
    model_save_path = f"models/gru_no_prev_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_save_path)
    with open("results/no_previous_act.json", "w", encoding="utf-8") as f:
        json.dump(results, f)

model_save_path = f"models/gru_best.pth"
torch.save(model.state_dict(), model_save_path)


