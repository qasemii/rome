import os, re, json
import torch, numpy
from collections import defaultdict
from itertools import combinations

from util import nethook
from util.globals import DATA_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    Qwen2ForCausalLM,
    Gemma2ForCausalLM,
    LlamaForCausalLM,
    OlmoForCausalLM,
)
from hf_olmo import OLMoForCausalLM
from experiments.utils import (
    ModelAndTokenizer,
    layername,
)
from experiments.utils import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
    make_noisy_embeddings,
    collect_token_range,
)
from dsets import KnownsDataset
from dsets.data_utils import check_whitespace

import random
import shutil
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from tqdm import tqdm
import json
from datasets import load_dataset
from pprint import pprint

random.seed(42)

torch.set_grad_enabled(False)

nltk.download('punkt_tab')


def get_rationales(mt, prompt, mode='prob', window=1):
    inp = make_inputs(mt.tokenizer, [prompt] * 11)
    with torch.no_grad():
        base_scores = predict_from_input(mt.model, inp)[0]

    answer = predict_token(mt, [prompt])[0]
    answer_t = mt.tokenizer.encode(answer)[0]
    base_score = base_scores[answer_t]

    tokens = nltk.word_tokenize(prompt)

    tokens = ['"' if token in ['``', "''"] else token for token in tokens]
    tokens = check_whitespace(prompt, tokens)
    tokens_range = collect_token_range(mt, prompt, 1)
    if window > 1:
        tokens_range = list(combinations(tokens_range, window))

    score_table = torch.zeros(len(tokens_range), len(inp['input_ids'][0]))
    for idx, t_range in enumerate(tokens_range):
        t_range = [t_range] if window == 1 else list(t_range)

        high = 10.0
        low = 0
        epsilon = 0.001

        while (high - low > epsilon):  # Sufficient iterations for precision
            noise = (low + high) / 2
            with torch.no_grad():
                low_scores = make_noisy_embeddings(mt, inp, tokens_to_mix=t_range, noise=noise)

            prob = low_scores[answer_t]
            # rank = torch.sort(low_scores, dim=-1, descending=True).indices.tolist().index(answer_t)
            rank = (low_scores > prob).sum().item()
            if rank == 0:
                low = noise
            else:
                high = noise

        print(t_range, noise, rank)
        if mode == 'noise':
            score = 10.0 - noise
        elif mode == 'prob':
            score = (base_score - prob)
        else:
            raise ValueError(f'Invalid mode: {mode}')

        for (b, e) in t_range:
            score_table[idx, b:e] = score

    token_range = collect_token_range(mt, prompt, 1)

    token_scores = torch.sum(score_table, dim=0) / torch.sum((score_table != 0), dim=0)

    word_scores = torch.tensor([token_scores[t_range[0]].item() for t_range in token_range])
    word_scores = torch.softmax(word_scores, dim=-1)

    results = {}

    results['input_ids'] = inp["input_ids"][0]
    results['input_tokens'] = tokens
    results['answer'] = answer
    results['base_score'] = base_score

    results['word_scores'] = word_scores.to(mt.model.device)
    results['token_scores'] = token_scores.unsqueeze(dim=0).to(mt.model.device)

    return results
