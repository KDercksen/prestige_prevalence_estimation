#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from annoy import AnnoyIndex
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from tqdm import tqdm

from utils import read_jsonl, write_json


def make_transformer_embeddings(texts, t_emb, normalize=True):
    s = [Sentence(x["text"]) for x in texts]
    t_emb.embed(s)
    e = [x.embedding.cpu().detach().numpy() for x in s]
    if normalize:
        return [x / (np.linalg.norm(x) + 1e-8) for x in e]
    else:
        return e


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=69)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--numtrees", type=int, default=10)
    p.add_argument("--metric", type=str, default="angular")
    p.add_argument("--data", type=str)
    p.add_argument("--transformer", type=str)
    p.add_argument("--outfile", type=str)
    p.add_argument("--batchsize", type=int, default=4)
    p.add_argument("--subsample", type=float)
    p.add_argument("--normalize", action="store_true")
    args = p.parse_args()

    all_sents = read_jsonl(args.data)
    t_emb = TransformerDocumentEmbeddings(args.transformer)
    index = AnnoyIndex(args.dim, args.metric)

    np.random.seed(args.seed)

    if args.subsample:
        print("Filtering reports for subsample...")
        num_reports = all_sents[-1]["report_idx"]
        idxs = np.where(np.random.random(num_reports) < args.subsample)[0]
        new_all_sents = []
        for sent in tqdm(all_sents):
            if sent["report_idx"] in idxs:
                new_all_sents.append(sent)
        all_sents = new_all_sents

    print("Starting index build...")
    sent_map = {}
    for i in tqdm(range(0, len(all_sents), args.batchsize)):
        sents = all_sents[i : i + args.batchsize]
        embeddings = make_transformer_embeddings(sents, t_emb, args.normalize)
        for x in range(min(args.batchsize, len(embeddings))):
            index.add_item(i + x, embeddings[x])
            sent_map[i + x] = sents[x]

    index.build(args.numtrees)
    index.save(args.outfile)
    write_json(args.outfile + "-sent_map", sent_map)
