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
from experiments.utils import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
    plot_all_flow,

    make_noisy_embeddings,
    trace_important_states,
    trace_important_window,
    calculate_noisy_result,
    plot_aggregated_heatmap,
    plot_hidden_aggregation,
    plot_hidden_flow,
    plot_all_flow
)
from dsets import KnownsDataset

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


def get_noisy_score(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    expect=None,
    topk=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answers_t, base_scores = [d[0] for d in predict_from_input(mt.model, inp, topk=topk)]

    answers = decode_tokens(mt.tokenizer, answers_t)
    answers = [a.strip() for a in answers]

    if expect is not None:
        if topk is None:
            raise ValueError("topk is None.")
        if not expect in answers:
            raise ValueError(f"'{expect}' is not in top-{topk} predictions.")
        index = answers.index(expect)
        answer_t = answers_t[index]
        base_score = base_scores[index]
    else:
        answer_t = answers_t[0]
        base_score = base_scores[0]

    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    try:
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    except:
        print(f"Couldn't find any token range for {subject}.")
        print("Attempting dummy token range ...")
        e_range = (-2, -1)
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    low_score, rank = make_noisy_embeddings(
        mt.model, inp, [], answer_t, e_range, noise=noise, uniform_noise=uniform_noise
    )

    low_score = low_score.item()
    return dict(
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        answer=answer,
        low_rank=rank,
    )

def extract_rationales(
    mt,
    prompt,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    normalize=False,
    kind=None,
    expect=None,
    topk=None,
    snippet_to_corrupt=None,
):
    main_score = predict_token(
        mt,
        [prompt],
        return_p=True,
    )[1][0].cpu()

    # Tokenize sentence into words and punctuation
    if snippet_to_corrupt:
      tokens = nltk.word_tokenize(snippet_to_corrupt)
    else:
      tokens = nltk.word_tokenize(prompt)

    results = {}
    high_score = list()
    low_score = list()
    low_rank = list()
    high_rank = list()
    score = list()

    for word in tokens:
        flow = calculate_noisy_result(
            mt,
            prompt,
            token=word,
            samples=samples,
            noise=noise,
            uniform_noise=uniform_noise,
            window=window,
            kind=kind,
            expect=expect,
            topk=topk,
        )

        # indirect score
        # OPTION 1
        # ms = torch.topk(flow['scores'].flatten(), 3).values
        # ms = torch.sum(torch.sum(ms))

        # OPTION 2
        # ms = torch.mean(flow['scores'])

        # OPTION 3
        imax = torch.argmax(flow['scores'].flatten()).item()
        ms = flow['scores'].flatten()[imax]

        # OPTION 4

        # window_s = max(0,imax-5)
        # window_e = min(flow['scores'].numel(), window_s+10)
        # ms = torch.sum(torch.sum(flow['scores'].flatten()[imax]))

        high_score.append(ms)

        ls = flow['low_score'] # low score
        low_score.append(ls)

        # low rank
        lr = flow['low_rank']
        low_rank.append(lr+1)

        # high rank
        hr = flow['ranks'].flatten()[imax]
        high_rank.append(lr+1)

        s = ms - ls #/(lr+1)
        score.append(s)

    for k, v in flow.items():
        results[k] = v

    results['main_score'] = main_score
    results['high_score'] = torch.tensor(high_score)
    results['low_score'] = torch.tensor(low_score)
    results['low_rank'] = torch.tensor(low_rank)
    results['high_rank'] = torch.tensor(high_rank)
    results['scores'] = torch.tensor(score).unsqueeze(dim=0)
    # breakpoint()

    if normalize:
      results['scores'] = torch.softmax(results['scores'], dim=1)

    results['input_tokens'] = tokens

    return results

    # plot_rationales(results, savepdf)


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
        ax.annotate(label, (0.5 + i, 1.0), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, rotation=0)

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
    fig_width = cell_width * (max_length-min_length-1)
    fig_height = cell_height * min_length


    # Create the heatmap
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(np.transpose(scores), annot=True, fmt=".2f", cmap="Greens", xticklabels=x_labels, yticklabels=y_labels, cbar=False)

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