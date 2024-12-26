import os
import json
import re

import pandas as pd

FOLDER_PATH = "E:\\task-oriented-chatbot\\multiwoz\\data\\MultiWOZ_2.2"
DIALOGUE_ID = 'dialogue_id'
SERVICES = 'services'
TURNS = 'turns'
FRAMES = "frames"
SPEAKER = "speaker"
TURN_ID = "turn_id"
UTTERANCE = "utterance"
ACTIONS = "actions"
SLOTS = "slots"
STATE = "state"
SERVICE = "service"
ACTIVE_INTENT = "active_intent"

list_files = os.listdir(os.path.join(FOLDER_PATH, 'test'))
dialogues = []
for filename in list_files:
    with open(os.path.join(FOLDER_PATH, 'dev', filename), 'r', encoding="utf-8") as f:
        data = json.load(f)
        dialogues.extend(data)

print(len(dialogues))

with open(os.path.join(FOLDER_PATH, "dialog_acts.json"), "r", encoding="utf-8") as f:
    acts = json.load(f)


def tokenize_with_offsets(sentence: str):
    tokens = []
    offsets = []

    pattern = re.finditer(r'\S+', sentence)
    for match in pattern:
        token = match.group()
        start = match.start()
        end = match.end() - 1
        tokens.append(token)
        offsets.append((start, end))

    return tokens, offsets


dialogue_ids = []
utterances = []
speakers = []
services = []
turn_ids = []
bio_tags = []
active_intents = []
d_acts = []

for dialogue in dialogues:
    # print("-" * 25)
    # print(dialogue[DIALOGUE_ID])
    dialog_acts = acts[dialogue[DIALOGUE_ID]]
    last_domain = ""
    for turn in dialogue[TURNS]:
        dialogue_ids.append(dialogue[DIALOGUE_ID])
        utterances.append(turn[UTTERANCE])
        turn_ids.append(turn[TURN_ID])
        speakers.append(turn[SPEAKER])
        # print(f"Turn {turn[TURN_ID]} {turn[SPEAKER]}: {turn[UTTERANCE]}")
        frames = turn[FRAMES]
        # if dialogue[DIALOGUE_ID] == "MUL0160.json":
        #     print(frames)

        match len(frames):
            case 0:
                active_intents.append("NONE")
            case 1:
                active_intents.append("NONE")
            case 2:
                frames_has_state = [frame for frame in frames if STATE in frame]
                intent_list = [frame[STATE][ACTIVE_INTENT] for frame in frames_has_state
                               if frame[STATE][ACTIVE_INTENT] != "NONE"]
                if len(intent_list) == 0:
                    active_intents.append("NONE")
                else:
                    active_intents.append(intent_list[0])

            case 8:
                frames_has_state = [frame for frame in frames if STATE in frame]

                intent_list = [frame[STATE][ACTIVE_INTENT] for frame in frames_has_state
                               if frame[STATE][ACTIVE_INTENT] != "NONE"]
                if len(intent_list) == 0:
                    active_intents.append("NONE")
                else:
                    active_intents.append(intent_list[0])

            case _:
                active_intents.append("NONE")

        turn_acts = dialog_acts[turn[TURN_ID]]['dialog_act']
        act_keys = [turn_act for turn_act in turn_acts.keys()]
        if len(act_keys) != 0:
            d_acts.append(act_keys[0])
        else:
            d_acts.append("NONE")

        tokens, offsets = tokenize_with_offsets(turn[UTTERANCE])
        span_info = dialog_acts[turn[TURN_ID]]['span_info']
        bio_tag = ['O'] * len(tokens)

        for span in span_info:
            slot = span[1]
            start, end = span[3], span[4]

            for i, (token_start, token_end) in enumerate(offsets):
                if token_start >= start and token_end <= end:
                    if token_start == start:
                        if bio_tag[i] == 'O':
                            bio_tag[i] = f"B-{slot}"
                        else:
                            pass
                    else:
                        if bio_tag[i] == 'O':
                            bio_tag[i] = f"I-{slot}"
                        else:
                            pass
        bio_tags.append(bio_tag)

        # print()

print(len(active_intents))

d = {'dialogue_id': dialogue_ids,
     'turn': turn_ids,
     'utterance': utterances,
     'bio_tag': bio_tags,
     'dialogue_act': d_acts,
     'speaker': speakers,
     'active_intent': active_intents}

df = pd.DataFrame(data=d)
df.to_csv("transformed_data/test/test.csv")
