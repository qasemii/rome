import argparse
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
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
    plot_all_flow,

    trace_with_patch,
    trace_important_states,
    trace_important_window,
    calculate_hidden_flow,
    plot_aggregated_heatmap,
    plot_hidden_aggregation,
    plot_hidden_flow,
    plot_all_flow
)

from experiments.rationalization import (
    extract_rationales,
    plot_rationales,
    plot_generation
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


def main():
    parser = argparse.ArgumentParser(description="Rationalization")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--model_name",
        default="gpt2-medium",
        choices=[
            "EleutherAI/gpt-j-6B",
            "EleutherAI/gpt-neox-20b",
            "gpt2-xl",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
        ],
    )
    aa("--kind", default="mlp", type=str)
    aa("--fact_file", default=None)
    aa("--output_dir", default=f"results/")
    aa("--noise_level", default=None, type=float)
    aa("--replace", default=0, type=int)
    args = parser.parse_args()

    nltk.download('punkt_tab')

    result_dir = f"{args.output_dir}/{args.model_name}"
    os.makedirs(result_dir, exist_ok=True)

    mt = ModelAndTokenizer(
        args.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        adapter_name_or_path=None)

    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)
    else:
        with open(args.fact_file) as f:
            knowns = json.load(f)

    uniform_noise = False

    if args.model_name == "gpt2":
        base_noise_level = 0.1346435546875
    elif args.model_name == "gpt2-medium":
        base_noise_level = 0.10894775390625
    elif args.model_name == "gpt2-large":
        base_noise_level = 0.053924560546875
    elif args.model_name == "gpt2-xl":
        base_noise_level = 0.0450439453125
    elif args.model_name == "EleutherAI/gpt-j-6B":
        base_noise_level = 0.031341552734375
    else:
        raise ValueError
    noise_level = 3 * base_noise_level

    predictions = []
    for knowledge in tqdm(knowns):
        p = predict_token(
            mt,
            [knowledge["prompt"]], # original/relevant/irrelevant/counterfact
            return_p=True,
            topk=10
        )
        predictions.append(p)

    org_true = [i for i, r in enumerate(results) if results[i][0][0].strip() == true_answers[i]]
    print('Number of True predictions: ', len(org_true))

    print("Starting rationalization ...")
    for idx in tqdm(org_true):
        knowledge = knowns[idx]
        known_id = knowledge["known_id"]
        filename = f"{result_dir}/{known_id}.npz"
        if not os.path.isfile(filename):
            result = extract_rationales(
                mt,
                knowledge["prompt"],
                knowledge["subject"],
                expect=knowledge["attribute"],
                topk=20,
                kind=args.kind,
                noise=noise_level,
                uniform_noise=uniform_noise,
            )
            save(result, filename)
        else:
            continue

def save(data, dir = None):
    with open(f'{dir}.pkl', 'wb') as file:
        pickle.dump(data, file)

def load(dir=None):
    with open(f'{dir}.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

if __name__ == "__main__":
    main()



