import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)

from dsets import KnownsDataset, CounterFactDataset

import random
import shutil
import nltk
import numpy as np
import pickle
from tqdm import tqdm

def save(data, dir=None):
    with open(f'{dir}.pkl', 'wb') as file:
        pickle.dump(data, file)

def load(dir=None):
    with open(f'{dir}.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def get_predictions(mt, data, topk=10):
    results = []
    for d in tqdm(data):
        predictions = predict_token(
            mt,
            [d["prompt"]],
            return_p=True,
            topk=topk
        )
        results.append(predictions)

    return results


def match_tokens_with_scores(mt, mem_ers):

    test = []
    tokenizer = mt.tokenizer
    scores = mem_ers['scores'].squeeze()

    for i, token in enumerate(mem_ers['input_tokens']):
        token = f" {token}" if i >= 0 else token  # Adding space if index is valid
        encoded_token = tokenizer.encode(token)
        token_length = len(encoded_token)
        test.extend([scores[i].item()] * token_length)

    return torch.tensor(test).unsqueeze(dim=0)