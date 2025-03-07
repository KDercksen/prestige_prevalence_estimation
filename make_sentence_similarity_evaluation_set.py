#!/usr/bin/env python
# -*- coding: utf-8 -*-

from annoy import AnnoyIndex
import pandas as pd
from tqdm import tqdm
from utils import read_json
from rich import print as rprint
import numpy as np

index = AnnoyIndex(768, "angular")
index.load("./doc_bert_nli_full_index.ann")

rng = np.random.default_rng(42)

bins = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
bin_counts = np.array([0, 0, 0, 0, 0])

pairs = []
for i in range(bins.shape[0] - 1):
    bin_low = bins[i]
    bin_high = bins[i + 1]
    while bin_counts[i] < 200:
        sent1, sent2 = rng.integers(0, index.get_n_items(), 2)
        d = index.get_distance(sent1, sent2)
        if bin_low <= (sim := ((2 - d) / 2)) <= bin_high and sent1 != sent2:
            bin_counts[i] += 1
            rprint(bin_counts)
            pairs.append((sent1, sent2, sim))

sentences = read_json("doc_bert_full_index.ann-sent_map")
data = []
for pair in pairs:
    data.append(
        (
            sentences[str(pair[0])]["text"],
            sentences[str(pair[1])]["text"],
            pair[2],
        )
    )

df = pd.DataFrame(data)
df.to_csv(
    "data/prevalence_sentence_sim_for_annotation_v2.csv", index=False, header=False
)
