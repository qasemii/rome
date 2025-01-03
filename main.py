import argparse
import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
)
from experiments.causal_trace import (
    predict_token,
)
from experiments.rationalization import (
    extract_rationales,
)

from dsets import (
    KnownsDataset,
    CounterFactDataset,
)
from dsets.data_utils import match_tokens_with_scores

import random
import shutil
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

from ReAGent.src.rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
from ReAGent.src.rationalization.rationalizer.importance_score_evaluator.delta_prob import \
    DeltaProbImportanceScoreEvaluator
from ReAGent.src.rationalization.rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from ReAGent.src.rationalization.rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from ReAGent.src.rationalization.rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler

from ReAGent.src.evaluation.evaluator.soft_norm_sufficiency import SoftNormalizedSufficiencyEvaluator
from ReAGent.src.evaluation.evaluator.soft_norm_comprehensiveness import SoftNormalizedComprehensivenessEvaluator

import csv

random.seed(42)

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser(description="Rationalization")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-medium")
    aa("--kind", default="mlp", type=str)
    aa("--fact_file", default="knowns")
    aa("--output_dir", default=f"results/")
    aa("--noise_level", default=None, type=float)
    aa("--method",
       type=str,
       default="integrated_gradients",
       help="membre, reagent, \
        attention, attention_last, attention_rollout, \
        gradient_shap, input_x_gradient, integrated_gradients, lime")  # TODO

    args = parser.parse_args()

    nltk.download('averaged_perceptron_tagger_eng')

    result_dir = f"{args.output_dir}{args.fact_file}/{args.model_name}"
    os.makedirs(result_dir, exist_ok=True)

    print('Loading model and tokenizer ...')
    mt = ModelAndTokenizer(
        args.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        )

    if args.fact_file == "knowns":
        dataset = KnownsDataset(DATA_DIR)
    elif args.fact_file == "counterfact":
        dataset = CounterFactDataset(DATA_DIR)
    else:
        raise ValueError

    uniform_noise = False

    # init rationalizer
    rational_size = 3
    rational_size_ratio = None

    # tested with 3 0.1 5000 5
    stopping_top_k = 3
    replacing = 0.1
    max_step = 3000
    batch = 3

    print("Getting model's predictions...")
    predictions = []
    for data in tqdm(dataset):
        p = predict_token(
            mt,
            [data["prompt"]],  # original/relevant/irrelevant/counterfact
            return_p=True,
            topk=10
        )
        predictions.append(p)

    true_predictions_idx = [i for i, r in enumerate(predictions) if
                            predictions[i][0][0].strip() == dataset[i]['target']]
    print(f"Number of True predictions: {len(true_predictions_idx)}/{len(dataset)}")

    if args.method == 'membre':
        nltk.download('punkt_tab')

        if args.noise_level is None:
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
                raise ValueError("Please choose the right model for noise_level")
            noise_level = 3 * base_noise_level
        else:
            noise_level = args.noise_level

        kind = args.kind
    elif args.method == 'random':
        pass
    elif args.method == 'reagent':

        token_sampler = POSTagTokenSampler(tokenizer=mt.tokenizer, device=mt.model.device)

        stopping_condition_evaluator = TopKStoppingConditionEvaluator(
            model=mt.model,
            token_sampler=token_sampler,
            top_k=stopping_top_k,
            top_n=rational_size,
            top_n_ratio=rational_size_ratio,
            tokenizer=mt.tokenizer
        )
        importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
            model=mt.model,
            tokenizer=mt.tokenizer,
            token_replacer=UniformTokenReplacer(
                token_sampler=token_sampler,
                ratio=replacing
            ),
            stopping_condition_evaluator=stopping_condition_evaluator,
            max_steps=max_step
        )
        rationalizer = AggregateRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            batch_size=batch,
            overlap_threshold=2,
            overlap_strict_pos=True,
            top_n=rational_size,
            top_n_ratio=rational_size_ratio
        )
    elif args.method == 'attention_last' or args.method == 'attention_rollout':
        from ReAGent.src.rationalization.rationalizer.importance_score_evaluator.attention import \
            AttentionImportanceScoreEvaluator
        importance_score_evaluator = AttentionImportanceScoreEvaluator(
            model=mt.model,
            tokenizer=mt.tokenizer,
            attn_type=args.method.replace("attention_", "")
        )
        from ReAGent.src.rationalization.rationalizer.sample_rationalizer import SampleRationalizer
        rationalizer = SampleRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            top_n=3,
        )
    else:
        # assert args.method in ['integrated_gradients', 'input_x_gradient', 'attention', 'gradient_shap'] # input_x_gradient = signed in self written
        from ReAGent.src.rationalization.rationalizer.importance_score_evaluator.inseq import \
            InseqImportanceScoreEvaluator
        importance_score_evaluator = InseqImportanceScoreEvaluator(
            model=mt.model,
            tokenizer=mt.tokenizer,
            method=args.method,  # integrated_gradients input_x_gradient attention
            attribute_params={
            }
        )
        from ReAGent.src.rationalization.rationalizer.sample_rationalizer import SampleRationalizer
        rationalizer = SampleRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            top_n=3,
        )

    # init evaluator
    soft_norm_suff_evaluator = SoftNormalizedSufficiencyEvaluator(mt.model)
    soft_norm_comp_evaluator = SoftNormalizedComprehensivenessEvaluator(mt.model)

    print("Starting rationalization ...")
    results = {}
    for idx in tqdm(true_predictions_idx[:100]):
        results[idx] = {}
        data = dataset[idx]
        filename = f"{result_dir}/{data['id']}.pkl"

        input_ids = mt.tokenizer(data["prompt"], return_tensors='pt')['input_ids'][0].to(mt.model.device)
        target_id = mt.tokenizer((" "+data["target"]), return_tensors='pt')['input_ids'][0].squeeze(dim=0).to(mt.model.device)

        if args.method == 'membre':
            ers = extract_rationales(
                mt,
                data["prompt"],
                kind=kind,
                noise=noise_level,
                uniform_noise=uniform_noise,
            )
            # save(ers, filename)
            # breakpoint()
            scores = match_tokens_with_scores(mt, data=data, ers=ers).to(mt.model.device)
            results[idx]["membre"] = ers
        elif args.method == 'random':
            scores = torch.softmax(torch.rand(input_ids.unsqueeze(dim=0).shape, device=mt.model.device), dim=-1)
        else:
            # rationalization
            rationalizer.rationalize(input_ids.unsqueeze(dim=0), target_id.unsqueeze(dim=0))
            scores = rationalizer.mean_important_score.unsqueeze(dim=0).to(mt.model.device)

        input_ids_step = torch.unsqueeze(input_ids, 0)
        target_id_step = torch.unsqueeze(target_id, 0)

        # importance score by Random Score
        # random_scores = torch.softmax(torch.rand(scores.shape, device=mt.model.device), dim=-1)
        try:
            # compute Soft-NS and Soft-NC on source importance score
            source_soft_ns_step = soft_norm_suff_evaluator.evaluate(input_ids_step, target_id_step, scores)
            source_soft_nc_step = soft_norm_comp_evaluator.evaluate(input_ids_step, target_id_step, scores)
            print(f"Source Soft-NS: {source_soft_ns_step}, Source Soft-NC: {source_soft_nc_step}")

            # # compute Soft-NS and Soft-NC on random importance score
            # random_soft_ns_step = soft_norm_suff_evaluator.evaluate(input_ids_step, target_id_step, random_scores)
            # random_soft_nc_step = soft_norm_comp_evaluator.evaluate(input_ids_step, target_id_step, random_scores)
            # print(f"Random Soft-NS: {random_soft_ns_step}, Random Soft-NC: {random_soft_nc_step}")

            # # compute metrics on Soft-NS and Soft-NC
            # metric_soft_ns = torch.log(source_soft_ns_step / random_soft_ns_step)
            # metric_soft_nc = torch.log(source_soft_nc_step / random_soft_nc_step)
            # print(f"metric_soft_ns: {metric_soft_ns}, metric_soft_nc: {metric_soft_nc}")


            results[idx] = {'scores': scores.squeeze(),
                            'source_soft_ns': source_soft_ns_step.item(),
                            'source_soft_nc': source_soft_nc_step.item(),}
                            # 'random_soft_ns': random_soft_ns_step.item(),
                            # 'random_soft_nc': random_soft_nc_step.item(),}
        except:
            print(f"Unable to get the score for {idx}")
            continue
    # export results
    Path(result_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(result_dir, f'{args.method}.pkl'), 'wb') as outfile:
        pickle.dump(results, outfile)



if __name__ == "__main__":
    main()



