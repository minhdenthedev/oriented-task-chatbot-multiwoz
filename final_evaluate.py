import json
import pandas as pd
import ast

MODE = "train"

with open("results/decoded_" + MODE + "_outputs.json", "r", encoding="utf-8") as f:
    output = json.load(f)

data = pd.read_csv("transformed_data/" + MODE + "/" + MODE + ".csv", index_col=0)
data = data[data['speaker'] == "USER"]
data = data.reset_index()
data['bio_tag'] = data.apply(lambda row: ast.literal_eval(row['bio_tag']) + [row['active_intent']], axis=1)

bio_decoded = output['bio_decoded']
act_decoded = output['act_decoded']


def parse_slot_value_pair(tokens, tags):
    slot_value_pairs = []
    current_slot = None
    current_value = []

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if current_slot:
                slot_value_pairs.append({"slot": current_slot, "value": " ".join(current_value)})
            current_slot = tag[2:]
            current_value = [token]
        elif tag.startswith('I-') and current_slot == tag[2:]:
            current_value.append(token)
        else:
            if current_slot:
                slot_value_pairs.append({"slot": current_slot, "value": " ".join(current_value)})
                current_slot = None
                current_value = []

    # Handle the last slot-value pair
    if current_slot:
        slot_value_pairs.append({"slot": current_slot, "value": " ".join(current_value)})

    results = [f"{pair['slot']}={pair['value']}" for pair in slot_value_pairs]

    return results


def get_intent(tokens, tags):
    return tags[len(tokens)]


def evaluate(true, predicted, classes=None):
    if classes is None:
        classes = set(true) | set(predicted)

    per_class_metrics = {}

    for cls in classes:
        tp = sum((t == cls and p == cls) for t, p in zip(true, predicted))
        fp = sum((p == cls and t != cls) for t, p in zip(true, predicted))
        fn = sum((t == cls and p != cls) for t, p in zip(true, predicted))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}

    precision_avg = sum(metrics["precision"] for metrics in per_class_metrics.values()) / len(classes)
    recall_avg = sum(metrics["recall"] for metrics in per_class_metrics.values()) / len(classes)
    f1_avg = sum(metrics["f1"] for metrics in per_class_metrics.values()) / len(classes)

    return per_class_metrics, {"precision": precision_avg, "recall": recall_avg, "f1": f1_avg}


true_pairs = [parse_slot_value_pair(data.loc[i, 'utterance'].split(), data.loc[i, 'bio_tag'])
              for i in range(len(data))]
predicted_pairs = [parse_slot_value_pair(data.loc[i, 'utterance'].split(), bio_decoded[i])
                   for i in range(len(data))]

true_intents = [get_intent(data.loc[i, 'utterance'].split(), data.loc[i, 'bio_tag'])
                for i in range(len(data))]
predicted_intents = [get_intent(data.loc[i, 'utterance'].split(), bio_decoded[i])
                     for i in range(len(data))]

true_acts = data['dialogue_act'].tolist()
predicted_acts = act_decoded

_, result = evaluate(true_acts, predicted_acts)
print(f"Act f1 score: {result}")

_, result = evaluate(true_intents, predicted_intents)
print(f"Intent f1 score: {result}")

pair_correct = 0
all_pairs = 0
for i in range(len(true_pairs)):
    true = true_pairs[i]
    predict = predicted_pairs[i]
    all_pairs += max(len(true), len(predict))
    n = min(len(true), len(predict))
    for j in range(n):
        if true[j] == predict[j]:
            pair_correct += 1

print(f"Slot-value accuracy: {pair_correct/all_pairs}")


