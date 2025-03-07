#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer
import torch as t
import json
import torch.nn.functional as F
from annoy import AnnoyIndex
from tqdm import tqdm, trange

from utils import read_json, read_jsonl, write_json


def cache_vectors(sent_index):
    # first, cache all the vectors from the index
    vectors = t.zeros((sent_index.get_n_items(), 768), dtype=t.float16)
    for i in trange(sent_index.get_n_items(), desc="Caching embeddings to f16..."):
        vectors[i] = t.tensor(sent_index.get_item_vector(i), dtype=t.float16)
    return vectors


def make_global_prevalence_dataset(sent_index, sample_idxs, vectors, args):
    # calculate parwise cosine similarity in chunks and count the number of items for each row that are above the threshold
    counter_array = t.zeros((len(sample_idxs), len(args.thresholds)), dtype=int)
    bar = tqdm(total=len(sample_idxs))
    counter = 0
    for chunk_a in vectors[sample_idxs].chunk(1000):
        for chunk_b in vectors.chunk(10):
            chunk_a_norm = F.normalize(chunk_a, p=2, dim=1).to("cuda:0")
            chunk_b_norm = F.normalize(chunk_b, p=2, dim=1).to("cuda:0")
            sims = t.mm(chunk_a_norm, chunk_b_norm.transpose(0, 1))
            sims = (1 + sims) / 2
            assert sims.min() >= 0 and sims.max() <= 1
            for i, threshold in enumerate(args.thresholds):
                aggr_sims = t.sum(sims > threshold, dim=1).cpu()
                counter_array[counter : counter + chunk_a.size(0), i] += aggr_sims
        bar.update(len(sample_idxs) // 1000)
        counter += chunk_a.size(0)
    bar.close()
    similarities = counter_array.float() / sent_index.get_n_items()
    similarities = similarities.half()

    # choose random sample of findings to calculate similarities for
    store_similarities = similarities

    print("Saving intermediate results for global prevalence dataset...")
    t.save(store_similarities, args.save_directory / "global_prevalences_tensor.pkl")


def make_local_prevalence_dataset(
    vectors,
    all_sentences,
    sample_idxs,
    entity_annotations,
    args,
):
    report_idxs = list(dict.fromkeys([v["report_idx"] for v in all_sentences.values()]))

    report_idx_to_sents = defaultdict(list)
    for i, v in enumerate(all_sentences.values()):
        report_idx_to_sents[v["report_idx"]].append((i, v["text"]))

    # map to convert sent_map report idx to entity-annotations report idx...
    report_idx_map = dict(enumerate(report_idxs))
    report_idx_map_inv = dict((y, x) for x, y in enumerate(report_idxs))
    similarities = t.zeros(args.sample_size, len(args.thresholds), dtype=t.float16)

    le = MultiLabelBinarizer()
    encoded_labels = le.fit_transform(
        [report["entities"] for report in entity_annotations]
    ).astype(bool)

    report_comparison_cache = {}
    for i, idx in tqdm(
        enumerate(sample_idxs),
        desc="Building local prevalence dataset...",
        total=len(sample_idxs),
    ):
        sent = all_sentences[str(idx)]
        report_idx = report_idx_map_inv[sent["report_idx"]]
        local_entities = encoded_labels[report_idx]
        pairwise_jaccard_score = 1 - np.squeeze(
            pairwise_distances(
                local_entities.reshape(1, -1),
                encoded_labels,
                metric="jaccard",
                n_jobs=-1,
            )
        )
        if report_idx in report_comparison_cache:
            # check if the cache is already filled for this report
            compare_to = report_comparison_cache[report_idx]
        else:
            # if not, find unique reports to compare to based on jaccard sim
            compare_to = list(
                set(
                    [
                        report_idx_map[x]
                        for x in np.where(pairwise_jaccard_score > 0.3)[0]
                        if x != report_idx
                    ]
                )
            )
            report_comparison_cache[report_idx] = compare_to

        # compare_to contains report idxs of relevant reports. find all the corresponding sentences:
        compare_sents = [v[0] for x in compare_to for v in report_idx_to_sents[x]]

        # calculate the cosine similarity of vectors[idx] against vectors[compare_sents] in chunks of 10000
        # num_chunks = len(compare_sents) // 10000
        # sims = []
        # for chunk in vectors[compare_sents].chunk(num_chunks):
        #     sims.append(
        #         t.nn.functional.cosine_similarity(vectors[idx].view(1, -1), chunk)
        #     )
        # sims = t.cat(sims)
        # sims = t.nn.functional.cosine_similarity(
        #     vectors[idx].view(1, -1).to("cuda:0"), vectors[compare_sents].to("cuda:0")
        # ).cpu()
        # for j, threshold in enumerate(args.thresholds):
        #     if len(compare_to) == 0:
        #         similarities[i, j] = 0
        #     else:
        #         similarities[i, j] = t.sum(sims > threshold).item() / len(compare_sents)

    with open(args.save_directory / "comparison_cache.json", "w") as f:
        print(json.dumps(report_comparison_cache), file=f)

    # t.save(similarities, args.save_directory / "local_prevalences_tensor.pkl")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--annoy_index", type=Path, required=True)
    p.add_argument("--entity_annotations", type=Path, required=True)
    p.add_argument("--sample_size", type=int)
    p.add_argument("--sample_idxs", type=Path)
    p.add_argument("--save_directory", type=Path, required=True)
    p.add_argument("--thresholds", type=list, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--do_global", action="store_true")
    p.add_argument("--do_local", action="store_true")
    args = p.parse_args()

    args.save_directory.mkdir(parents=True, exist_ok=True)

    write_json(
        str(args.save_directory / "args.json"),
        {k: str(v) for k, v in vars(args).items()},
    )

    sent_index = AnnoyIndex(768, "angular")
    sent_index.load(str(args.annoy_index))
    all_sentences = read_json(str(args.annoy_index) + "-sent_map")
    entity_annotations = read_jsonl("./data/all_reports_v4_entities.json")

    if args.sample_size:
        sample_idxs = np.random.choice(
            np.arange(sent_index.get_n_items()), size=args.sample_size
        )
        write_json(args.save_directory / "sample_idxs.json", sample_idxs.tolist())
    elif args.sample_idxs:
        sample_idxs = read_json(args.sample_idxs)
    else:
        print("you have to specify either sample_size or sample_idxs file")
        exit()

    write_json(
        args.save_directory / "sentences.json",
        [all_sentences[str(i)] for i in sample_idxs],
    )
    vectors = cache_vectors(sent_index)
    if args.do_global:
        print("running global prevalence calculations...")
        make_global_prevalence_dataset(sent_index, sample_idxs, vectors, args)
    if args.do_local:
        print("running local prevalence calculations...")
        make_local_prevalence_dataset(
            vectors,
            all_sentences,
            sample_idxs,
            entity_annotations,
            args,
        )
