import torch
import torch.nn as nn
import torch.nn.functional as F


class MyGRU(nn.Module):
    def __init__(self, num_acts, num_tags, vocab_size):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=512, padding_idx=0)

        self.flatten = nn.Flatten()
        self.grus = nn.GRU(input_size=512, num_layers=3, hidden_size=256, dropout=0.2, batch_first=True)

        self.bio_head = nn.Linear(in_features=256, out_features=num_tags)

        self.classification_head = nn.Linear(in_features=274, out_features=128)
        self.dense_2 = nn.Linear(in_features=128, out_features=64)
        self.dense_3 = nn.Linear(in_features=64, out_features=num_acts)

    def forward(self, x, previous_acts):
        x = self.embeddings(x)
        gru_out, _ = self.grus(x)
        bio_logits = self.bio_head(gru_out)

        flattened = self.flatten(gru_out[:, -1, :])
        flattened = torch.cat((flattened, previous_acts), dim=1)
        # print(flattened.shape)
        classification_logits = self.classification_head(flattened)
        classification_logits = F.relu(classification_logits)
        classification_logits = F.dropout(classification_logits, p=0.2)
        classification_logits = self.dense_2(classification_logits)
        classification_logits = F.relu(classification_logits)
        classification_logits = F.dropout(classification_logits, p=0.2)
        classification_logits = self.dense_3(classification_logits)

        return bio_logits, classification_logits


# NUM_ACTS = 18
# NUM_TAGS = 47
# VOCAB_SIZE = 7752
# model = MyGRU(NUM_ACTS, NUM_TAGS, VOCAB_SIZE)
#
# print(model)