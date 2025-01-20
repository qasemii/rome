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


def extract_rationales(
        mt,
        prompt,
        samples=10,
        noise=0.1,
        window=1,
        uniform_noise=False,
        mode=None,
        normalize=False,
):
    # Llama and Gemma add bos token
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))  # [(samples + 1), 1]
    with torch.no_grad():
        base_scores = predict_from_input(mt.model, inp)[0]

    answer = predict_token(mt, [prompt])[0]
    answer_t = mt.tokenizer.encode(answer)[0]
    base_score = base_scores[answer_t]

    noised_scores = make_noisy_embeddings(
        mt, inp,
        tokens_to_mix=[(0, len(inp))],
        noise=noise,
    )
    noised_score = noised_scores[answer_t]

    results = {}
    token_scores = [0] if isinstance(mt.model, Gemma2ForCausalLM) or isinstance(mt.model, LlamaForCausalLM) else []

    # Tokenize sentence into words and punctuation
    tokens = nltk.word_tokenize(prompt)
    tokens = ['"' if token in ['``', "''"] else token for token in tokens]
    tokens = check_whitespace(prompt, tokens)

    tokens_range = collect_token_range(mt, prompt, 1)
    if window >= 1:
        tokens_range = list(combinations(tokens_range, window))
    score_table = torch.zeros(len(tokens_range), len(tokens))
    for i, r_list in enumerate(tokens_range):
        r_list = list(r_list)
        # breakpoint()
        try:
            suff_scores = make_noisy_embeddings(
                mt, inp, tokens_to_mix=r_list, noise=noise, uniform_noise=uniform_noise, denoise=True
            )

            comp_scores = make_noisy_embeddings(
                mt, inp, tokens_to_mix=r_list, noise=noise, uniform_noise=uniform_noise
            )
        except:
            print(f"Couldn't compute the low_scores. Assigning 0 to lower_score ...")
            low_scores = torch.zeros(mt.tokenizer.vocab_size, device=mt.model.device)

        if mode is None:
            suff_score = suff_scores[answer_t]
            comp_score = comp_scores[answer_t]
            score = (base_score - comp_score) + (suff_score - noised_score)
        else:
            topk = 10
            base_scores_topk = torch.topk(base_scores, topk).values
            topk_idx = torch.topk(base_scores, topk).indices
            low_scores_topk = low_scores.gather(0, topk_idx)

            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            score = kl_loss(low_scores_topk, base_scores_topk)

        for (b, e) in r_list:
            score_table[i, b:e] = score.item()

    # breakpoint()
    tokens_range = collect_token_range(mt, prompt, 1)
    word_scores = torch.sum(score_table, dim=0) / torch.sum((score_table != 0), dim=0)

    for i, r in enumerate(tokens_range):
        n_extend = r[1] - r[0]
        token_scores.extend([word_scores[i].item()] * n_extend)

    results['input_ids'] = inp["input_ids"][0]
    results['input_tokens'] = tokens
    results['answer'] = answer
    results['base_score'] = base_score
    results['word_scores'] = word_scores
    results['token_scores'] = torch.tensor(token_scores, device=mt.model.device).unsqueeze(dim=0)

    if normalize:
        results['scores'] = torch.softmax(results['scores'], dim=1)

    return results
