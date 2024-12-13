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


def get_dataset(data_name="knowns"):
    if data_name=="knowns":
        dataset = KnownsDataset(DATA_DIR)
    elif data_name="counterfact":
        dataset = CounterFactDataset(DATA_DIR)
    else:
        raise ValueError


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