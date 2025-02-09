import argparse
import os
import torch
from transformers import (
    Gemma2ForCausalLM,
    LlamaForCausalLM,
)
from util.globals import DATA_DIR

from experiments.utils import (
    ModelAndTokenizer,
    predict_token,
)
from experiments.rationalization import (
    get_rationales,
)

from dsets import (
    KnownsDataset,
    CounterFactDataset,
)

import random
import nltk
import pickle
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI

from rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
from rationalization.rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from rationalization.rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from rationalization.rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from rationalization.rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler

from experiments.utils import check_whitespace

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
    aa("--norm", default='2')
    aa("--mode", default='prob')
    aa("--method", type=str, default="integrated_gradients",
       help="noiser, reagent, attention, attention_last, attention_rollout, \
             gradient_shap, input_x_gradient, integrated_gradients, lime")  # TODO
    aa("--openai_api_key", type=str, default=None)

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

    print("Starting rationalization ...")

    client = OpenAI(
        api_key = args.openai_api_key
    )

    INSTRUCTION = (
        "# Task:\n"
        "Given a prompt and its tokens, identify tokens in the prompt that are necessary for predicting the given label. "
        "Token are necessary if they meet the following criteria:\n"
        "1. If we keep them and exclude the rest from the prompt, you can predict the label with high cofidence.\n"
        "2. If we exclude any of the necessary tokens you cannot predict the label with high confidence.\n"
    )


    PROMPT = (
        "Prompt: {prompt}\n"
        "Tokens: {tokens}\n"
        "Label: {label}\n"
        "Plausible elements: "
    )

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

        results = []
        # Gemma and Llama add [bos] token which should be exclude from input prompt when
        start_pos = 1 if isinstance(mt.model, Gemma2ForCausalLM) or isinstance(mt.model, LlamaForCausalLM) else 0
        for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0]):
            tokens = nltk.word_tokenize(data["prompt"])
            tokens = ['"' if token in ['``', "''"] else token for token in tokens]
            tokens = check_whitespace(data["prompt"], tokens)

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

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": INSTRUCTION},
                          {"role": "user",  "content": PROMPT.format(prompt=data["prompt"], 
                                                                     tokens=tokens,
                                                                     label=data["target"])}],
                temperature=0.0
            )
            tokens = [t.strip() for t in tokens]
            plausible_tokens = response.choices[0].message.content
            plausibility_scores = 0
            for t in plausible_tokens:
                idx = tokens.index(t)
                plausibility_scores += scores[idx]

            results.append({'id': data["id"],
                            'prompt': data["prompt"],
                            'target': data["target"],
                            'input_tokens': tokens,
                            'tokens_score': scores,
                            'plausible_tokens': plausible_tokens,
                            'plausibility_scores': plausibility_scores})
    
    Path(result_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(result_dir, f'{args.method}.pkl'), 'wb') as outfile:
        pickle.dump(results, outfile)

        


if __name__ == "__main__":
    main()



