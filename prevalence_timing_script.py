#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from time import time
from random import shuffle
import numpy as np
from tqdm import tqdm

import torch as t
import json

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("kdercksen/bert-base-multilingual-cased-1024")
model = AutoModelForSequenceClassification.from_pretrained(
    "kdercksen/bert-base-multilingual-cased-1024"
)


def read_json(fname):
    with open(fname, "r") as f:
        return json.load(f)


def read_jsonl(fname):
    with open(fname, "r") as f:
        return [json.loads(line) for line in f]


samples = [v for k, v in read_json("nli_bert_0.1_index.ann-sent_map").items()]
shuffle(samples)
samples = [v["text"] for v in samples][:500]

times = []
model.to("cuda:2")

for sample in tqdm(samples):
    start = time()
    model(
        tokenizer(
            sample, truncation=True, padding=True, return_tensors="pt"
        ).input_ids.to("cuda:2")
    )
    times.append(time() - start)

print(f"Average time: {np.mean(times)}")
print(f"Standard deviation: {np.std(times)}")

# Load annotated entities and reports
# reports = read_jsonl("data/all_reports_v4_entities.json")
# shuffle(reports)
# samples = reports[:500]

# times = []
# tmp = 0
# for sample in tqdm(samples):
#     start = time()
#     local_entities = set(sample["entities"])
#     for rep in reports:
#         if (
#             len(set(rep["entities"]) & local_entities)
#             / (len(set(rep["entities"]) | local_entities) + 1e-8)
#             > 0.5
#         ):
#             tmp += 1
#     times.append(time() - start)

# print(f"Average time: {np.mean(times)}")
# print(f"Standard deviation: {np.std(times)}")
