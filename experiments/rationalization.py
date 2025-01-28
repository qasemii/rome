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


def get_rationales(mt, prompt, scale_limit=1, mode='prob', verbose=False):
    # Use single prompt instead of 11
    inp = make_inputs(mt.tokenizer, [prompt] * 11)
    device = mt.model.device

    with torch.no_grad():
        base_scores = predict_from_input(mt.model, inp)[0]

    answer = predict_token(mt, [prompt])[0]
    answer_t = mt.tokenizer.encode(answer)[0]
    base_score = base_scores[answer_t].item()  # Ensure it's a scalar

    tokens = nltk.word_tokenize(prompt)
    tokens = ['"' if token in ['``', "''"] else token for token in tokens]
    tokens = check_whitespace(prompt, tokens)
    tokens_range = collect_token_range(mt, prompt)

    # Initialize on correct device
    tokens_score = torch.zeros(len(inp['input_ids'][0]), device=device)

    for idx, t_range in enumerate(tokens_range):
        b, e = t_range
        high = scale_limit
        low = 0.0
        for _ in range(10):  # with 10 iteration the precision would be 2^(-10)
            noise = (low + high) / 2
            with torch.no_grad():
                low_scores = make_noisy_embeddings(mt, inp, tokens_to_mix=t_range, noise=noise)
            prob = low_scores[answer_t].item()

            sorted_indices = torch.argsort(low_scores, descending=True)
            rank = (sorted_indices == answer_t).nonzero(as_tuple=True)[0].item()

            if rank == 0:
                low = noise
            else:
                high = noise

        if verbose:
            print(f"Token Range: {t_range}, Noise: {noise:.4f}, Output prob: {prob}")

        if mode == 'noise':
            score = scale_limit - noise
        elif mode == 'prob':
            score = base_score - prob
        else:
            raise ValueError(f'Invalid mode: {mode}')

        # Assign score to all subword tokens in the range
        tokens_score[b:e] = score

    tokens_score = tokens_score/torch.sum(tokens_score)

    # Aggregate word scores by averaging sub-tokens scores
    word_scores = torch.tensor([
        tokens_score[b:e].sum().item() for b, e in tokens_range
    ], device=device)

    # Consider removing softmax if raw scores are preferred
    # word_scores = torch.softmax(word_scores, dim=-1)

    return {
        'input_ids': inp["input_ids"][0],
        'input_tokens': tokens,
        'answer': answer,
        'base_score': base_score,
        'word_scores': word_scores,
        'token_scores': tokens_score.unsqueeze(0)
    }
