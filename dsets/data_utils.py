import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.utils import (
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


def match_tokens_with_scores(mt, data, ers):

    test = []
    tokens = nltk.word_tokenize(data['prompt'])
    scores = ers['scores'].squeeze()

    for i, token in enumerate(tokens):
        if i != 0:
            token = " " + token  # Adding space if index is valid
        encoded_token = mt.tokenizer.encode(token)
        token_length = len(encoded_token)
        # breakpoint()
        test.extend([scores[i].item()] * token_length)

    return torch.tensor(test).unsqueeze(dim=0)


