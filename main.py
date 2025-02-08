import argparse
import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from transformers import (
    Qwen2ForCausalLM,
    Gemma2ForCausalLM,
    LlamaForCausalLM,
    OlmoForCausalLM,
)
from hf_olmo import OLMoForCausalLM
from util.globals import DATA_DIR

from experiments.utils import (
    ModelAndTokenizer,
    layername,
    collect_embedding_std,
)
from experiments.utils import (
    predict_token,
)
from experiments.rationalization import (
    get_rationales,
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

from rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
from rationalization.rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from rationalization.rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from rationalization.rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from rationalization.rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler

from rationalization.src.evaluation.evaluator.soft_norm_sufficiency import SoftNormalizedSufficiencyEvaluator
from rationalization.src.evaluation.evaluator.soft_norm_comprehensiveness import \
    SoftNormalizedComprehensivenessEvaluator

import csv

device = "cuda"

random.seed(42)
torch.manual_seed(42)
# torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser(description="Rationalization")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-medium")
    aa("--dataset", default="Knowns")
    aa("--output_dir", default=f"results/")
    aa("--n_samples", default=-1, type=int)
    aa("--max_new_tokens", default=1, type=int)
    aa("--norm", default='inf')
    aa("--mode", default='prob')
    aa("--method",
       type=str,
       default="integrated_gradients",
       help="noiser, reagent, attention, attention_last, attention_rollout, \
             gradient_shap, input_x_gradient, integrated_gradients, lime")  # TODO

    args = parser.parse_args()

    nltk.download('averaged_perceptron_tagger_eng')

    result_dir = f"{args.output_dir}{args.dataset}/{args.model_name}"
    os.makedirs(result_dir, exist_ok=True)

    cache_dir = f"cache/{args.model_name}"
    os.makedirs(cache_dir, exist_ok=True)

    print('Loading model and tokenizer ...')
    mt = ModelAndTokenizer(
        args.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    pad_token_id = mt.tokenizer.pad_token_id if mt.tokenizer.pad_token_id is not None else mt.tokenizer.eos_token_id

    print(f"Loading {args.dataset} dataset ...")
    if args.dataset == "Knowns":
        dataset = KnownsDataset(DATA_DIR)
    elif args.dataset == "Counterfact":
        dataset = CounterFactDataset(DATA_DIR)
    elif args.dataset == "LongRA":
        # dataset = CounterFactDataset(DATA_DIR)
        pass
    else:
        raise ValueError
    
    # Filter dataset to only include examples where the predicted token matches the target
    print(f"Filtering dataset ...")
    dataset = [
        d for i, d in enumerate(dataset) 
        if predict_token(mt, [d['prompt']], topk=1)[0][0].strip() == d['target']
    ]
    print(f"Filtered dataset to {len(dataset)} examples")

    
    # init rationalizer
    rational_size = 3
    rational_size_ratio = None

    # tested with 3 0.1 5000 5
    stopping_top_k = 3
    replacing = 0.1
    max_step = 3000
    batch = 3

    if args.method == 'noiser':
        nltk.download('punkt_tab')
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
        from rationalization.rationalizer.importance_score_evaluator.attention import \
            AttentionImportanceScoreEvaluator
        importance_score_evaluator = AttentionImportanceScoreEvaluator(
            model=mt.model,
            tokenizer=mt.tokenizer,
            attn_type=args.method.replace("attention_", "")
        )
        from rationalization.rationalizer.sample_rationalizer import SampleRationalizer
        rationalizer = SampleRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            top_n=3,
        )
    else:
        # assert args.method in ['integrated_gradients', 'input_x_gradient', 'attention', 'gradient_shap'] # input_x_gradient = signed in self written
        from rationalization.rationalizer.importance_score_evaluator.inseq import \
            InseqImportanceScoreEvaluator
        importance_score_evaluator = InseqImportanceScoreEvaluator(
            model=mt.model,
            tokenizer=mt.tokenizer,
            method=args.method,  # integrated_gradients input_x_gradient attention
            attribute_params={
            }
        )
        from rationalization.rationalizer.sample_rationalizer import SampleRationalizer
        rationalizer = SampleRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            top_n=3,
        )

    # init evaluator
    soft_norm_suff_evaluator = SoftNormalizedSufficiencyEvaluator(mt.model)
    soft_norm_comp_evaluator = SoftNormalizedComprehensivenessEvaluator(mt.model)

    source_soft_ns = []
    source_soft_nc = []
    print("Starting rationalization ...")

    samples = dataset if args.n_samples == -1 else random.choices(dataset, k=args.n_samples)
    for data in tqdm(samples):
        idx = data['id']

        input_ids = mt.tokenizer(data["prompt"], return_tensors='pt')['input_ids'][0].to(mt.model.device)
        attention_mask = mt.tokenizer(data["prompt"], return_tensors='pt')['attention_mask'][0].to(mt.model.device)

        generated_ids = mt.model.generate(input_ids=torch.unsqueeze(input_ids, 0),
                                          attention_mask=torch.unsqueeze(attention_mask, 0),
                                          max_new_tokens=args.max_new_tokens,
                                          do_sample=False,
                                          pad_token_id=pad_token_id)[0]

        # generated_texts = mt.tokenizer.decode(generated_ids) for token in generated_ids
        # print(f'generated full sequence --> {generated_texts}')

        # Gemma and Llama add [bos] token which should be exclude from input prompt when
        start_pos = 1 if isinstance(mt.model, Gemma2ForCausalLM) or isinstance(mt.model, LlamaForCausalLM) else 0
        for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0]):
            target_id = generated_ids[target_pos]

            if args.method == 'noiser':
                ers = get_rationales(mt,
                                     data["prompt"],
                                     norm=args.norm,
                                     mode=args.mode,)
                scores = ers['token_scores']
            elif args.method == 'random':
                scores = torch.softmax(
                    torch.rand(torch.unsqueeze(generated_ids[:target_pos], 0).shape, device=mt.model.device), dim=-1)
            else:
                rationalizer.rationalize(torch.unsqueeze(generated_ids[:target_pos], 0), torch.unsqueeze(target_id, 0))
                scores = rationalizer.mean_important_score.unsqueeze(dim=0).to(mt.model.device)

            # importance score by Random Score
            rand_scores = torch.softmax(
                torch.rand(torch.unsqueeze(generated_ids[:target_pos], 0).shape, device=mt.model.device), dim=-1)

            try:
                # compute Soft-NS and Soft-NC on source importance score
                source_soft_ns_step = soft_norm_suff_evaluator.evaluate(torch.unsqueeze(generated_ids[:target_pos], 0),
                                                                        torch.unsqueeze(target_id, 0), scores)
                source_soft_nc_step = soft_norm_comp_evaluator.evaluate(torch.unsqueeze(generated_ids[:target_pos], 0),
                                                                        torch.unsqueeze(target_id, 0), scores)
                # print(f"Source Soft-NS: {source_soft_ns_step}, Source Soft-NC: {source_soft_nc_step}")

                # compute Soft-NS and Soft-NC on random importance score
                random_soft_ns_step = soft_norm_suff_evaluator.evaluate(torch.unsqueeze(generated_ids[:target_pos], 0),
                                                                        torch.unsqueeze(target_id, 0), rand_scores)
                random_soft_nc_step = soft_norm_comp_evaluator.evaluate(torch.unsqueeze(generated_ids[:target_pos], 0),
                                                                        torch.unsqueeze(target_id, 0), rand_scores)
                # print(f"Random Soft-NS: {random_soft_ns_step}, Random Soft-NC: {random_soft_nc_step}")

                # # compute metrics on Soft-NS and Soft-NC
                # metric_soft_ns = torch.log(source_soft_ns_step / random_soft_ns_step)
                # metric_soft_nc = torch.log(source_soft_nc_step / random_soft_nc_step)
                # print(f"metric_soft_ns: {metric_soft_ns}, metric_soft_nc: {metric_soft_nc}")

                source_soft_ns.append(source_soft_ns_step.item())
                source_soft_nc.append(source_soft_nc_step.item())

            except:
                print(f"Unable to get the score for {idx}")
                continue

    results = {'source_soft_ns': source_soft_ns,
               'source_soft_nc': source_soft_nc}
    # 'random_soft_ns': random_soft_ns_step.item(),
    # 'random_soft_nc': random_soft_nc_step.item(),}
    # export results
    Path(result_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(result_dir, f'{args.method}.pkl'), 'wb') as outfile:
        pickle.dump(results, outfile)


if __name__ == "__main__":
    main()



