import matplotlib.pyplot as plt
import json
import numpy as np

with open("E:\\task-oriented-chatbot\\results\\with_previous_act.json", "r", encoding="utf-8") as f:
    w_prev = json.load(f)

with open("E:\\task-oriented-chatbot\\results\\no_previous_act.json", "r", encoding="utf-8") as f:
    n_prev = json.load(f)


train_w_prev = w_prev['train']
train_n_prev = n_prev['train']
dev_w_prev = w_prev['dev']
dev_n_prev = n_prev['dev']


# fig, axs = plt.subplots(1, 1)

plt.title("BIO loss")
plt.plot(np.arange(len(train_w_prev['bio'])), train_w_prev['bio'], label="Training loss")
plt.plot(np.arange(len(dev_w_prev['bio'])), dev_w_prev['bio'], color="orange", label="Validating loss")

# axs[1].set_title("Act loss: No previous act")
# axs[1].plot(np.arange(len(train_n_prev['act'])), train_n_prev['act'], label="Training loss")
# axs[1].plot(np.arange(len(dev_n_prev['act'])), dev_n_prev['act'], label="Validating loss")
plt.legend()

plt.tight_layout()
plt.show()
