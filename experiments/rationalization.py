import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    Qwen2ForCausalLM,
    Gemma2ForCausalLM,
    LlamaForCausalLM,
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

    # Tokenize sentence into words and punctuation
    tokens = nltk.word_tokenize(prompt)
    tokens = ['"' if token in ['``', "''"] else token for token in tokens]
    tokens = check_whitespace(prompt, tokens)

    results = {}
    noise_score, main_score = [], []
    scores = [0] if isinstance(mt.model, Gemma2ForCausalLM) or isinstance(mt.model, LlamaForCausalLM) else []
    search_start = 0
    for token in tokens:
        try:
            token_range = find_token_range(mt.tokenizer, inp["input_ids"][0], token, search_start)
            low_scores = make_noisy_embeddings(
                mt.model, inp, token_range, noise=noise, uniform_noise=uniform_noise
            )

        except:
            print(f"Couldn't find any token range for {token}. Assigning 0 to lower_score ...")
            low_scores = torch.zeros(mt.tokenizer.vocab_size, device=mt.model.device)
            token_range = None

        search_start = search_start + len(token)  # 1 is for whitespace

        if mode is None:
            low_score = low_scores[answer_t]
            score = base_score - low_score
        else:
            kl_loss = torch.nn.KLDivLoss(reduce=True)
            score = kl_loss(base_scores, low_scores)
            # breakpoint()
        noise_score.append(low_scores[answer_t])
        main_score.append(score.item())
        n_extend = token_range[1] - token_range[0]
        scores.extend([score.item()] * n_extend)

    results['input_ids'] = inp["input_ids"][0]
    results['input_tokens'] = tokens
    results['answer'] = answer
    results['base_score'] = base_score
    results['low_scores'] = torch.tensor(noise_score, device=mt.model.device)
    results['main_scores'] = torch.tensor(main_score, device=mt.model.device)
    results['scores'] = torch.tensor(scores, device=mt.model.device).unsqueeze(dim=0)
    # breakpoint()
    if normalize:
        results['scores'] = torch.softmax(results['scores'], dim=1)

    return results


def plot_rationales(result, topk=None, savepdf=None, modelname=None):
    differences = result['scores']
    if topk != None:
        indexes = torch.topk(result['scores'], topk).indices.squeeze().tolist()
        for i in range(differences.numel()):
            if i not in indexes:
                differences[0][i] = 0

    # low_score = result["low_score"]
    answer = result["answer"].strip()
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    # for i in range(*result["subject_range"]):
    #     labels[i] = labels[i] + "*"

    fig, ax = plt.subplots(figsize=(len(labels), 0.5), dpi=200)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
            kind
        ],
        # vmin=low_score #Setting the minimum value of the color bar to 0
        # vmax=1    # Setting the maximum value of the color bar to 1
    )

    ax.set_xticks([0.5 + i for i in range(len(labels))])
    ax.set_xticklabels(labels, rotation=20, ha='center', fontsize=8)  # , position=(0.1, 0))

    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(f"[{answer}]", rotation=0, labelpad=30, va='center')

    scores_formatted = [f'{x.item():.4f}' for x in differences[0]]
    for i, label in enumerate(scores_formatted):
        ax.annotate(label, (0.5 + i, 1.0), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8,
                    rotation=0)

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_generation(data, topk=None, savepdf=None, modelname=None):
    # Labels for rows and columns
    x_labels = list([result['answer'] for result in data.values()])
    last_key = list(data.keys())[-1]
    y_labels = list(data[last_key]["input_tokens"])

    # Dummy data for a 2x4 grid
    scores = [result['scores'][0].tolist() for result in data.values()]
    min_length = min(len(s) for s in scores)
    max_length = max(len(s) for s in scores)
    padded_scores = [s + [np.nan] * (max_length - len(s)) for s in scores]
    scores = np.array(padded_scores)

    cell_width = 1.5
    cell_height = 1
    fig_width = cell_width * (max_length - min_length - 1)
    fig_height = cell_height * min_length

    # Create the heatmap
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(np.transpose(scores), annot=True, fmt=".2f", cmap="Greens", xticklabels=x_labels,
                     yticklabels=y_labels, cbar=False)

    ax.axhline(y=5.9, color='black', linewidth=0.5)  # x-axis
    ax.axvline(x=0, color='black', linewidth=0.5)  # y-axis

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # Show the plot
    # plt.tight_layout()
    plt.show()

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()