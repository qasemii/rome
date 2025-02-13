import argparse
import os
import torch
import random
import nltk
import pickle
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from transformers import (
    Gemma2ForCausalLM,
    LlamaForCausalLM,
)
from util.globals import DATA_DIR

from experiments.utils import (
    ModelAndTokenizer,
    predict_token,
    collect_token_range,
)
from experiments.rationalization import (
    get_rationales,
)

from dsets import (
    KnownsDataset,
    CounterFactDataset,
)

from experiments.utils import check_whitespace
from dsets.data_utils import match_tokens_with_scores

from rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
from rationalization.rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from rationalization.rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from rationalization.rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from rationalization.rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler


device = "cuda"

random.seed(42)
torch.manual_seed(42)
# torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser(description="Rationalization")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="Qwen/Qwen2-0.5B")
    aa("--dataset", default="Knowns")
    aa("--output_dir", default=f"results/")
    aa("--n_samples", default=-1, type=int)
    aa("--max_new_tokens", default=1, type=int)
    aa("--norm", default='2')
    aa("--mode", default='prob')
    aa("--method",
       type=str,
       default="noiser",
       help="noiser,attention, attention_last, attention_rollout, \
             gradient_shap, input_x_gradient, integrated_gradients, lime, reagent, occlusion")  # TODO
    aa("--openai_api_key", type=str, default=None)
    aa("--topk", type=int, default=3)
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

    client = OpenAI(
            api_key = args.openai_api_key
        )
    
    INSTRUCTION = (
        "# Task:\n"
        "Given a set of words extracted from a prompt for a completion task, "
        "return a single word as the most probable completion for the unseen "
        "prompt without any explanation."
    )


    PROMPT = (
        "Tokens: {tokens}\n"
        "Probable Completion: "
    )
    
    n_true_predictions = 0
    plaus_scores = []
    results = []
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
        # generated_texts = mt.tokenizer.decode(generated_ids)
        # print(f'generated full sequence --> {generated_texts}')

        scores_dict = {}
        # Gemma and Llama add [bos] token which should be exclude from input prompt when
        start_pos = 1 if isinstance(mt.model, Gemma2ForCausalLM) or isinstance(mt.model, LlamaForCausalLM) else 0
        for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0]):
            tokens = nltk.word_tokenize(data["prompt"])
            tokens = ['"' if token in ['``', "''"] else token for token in tokens]
            tokens = check_whitespace(data["prompt"], tokens)
            tokens_range = collect_token_range(mt, data["prompt"], tokens)

            target_id = generated_ids[target_pos]

            if args.method == 'noiser':
                ers = get_rationales(mt,
                                     data["prompt"],
                                     norm=args.norm,
                                     mode=args.mode,)
                scores = ers['word_scores']
            elif args.method == 'random':
                scores = torch.softmax(
                    torch.rand((len(tokens)), device=mt.model.device), dim=-1)
            else:
                rationalizer.rationalize(torch.unsqueeze(generated_ids[:target_pos], 0), torch.unsqueeze(target_id, 0))
                scores = rationalizer.mean_important_score.unsqueeze(dim=0).to(mt.model.device)
                scores = match_tokens_with_scores(scores.squeeze(), tokens_range)
            
            k = args.topk * len(tokens) // 100
            topk_indices = torch.topk(scores, k=k).indices.sort().values
            topk_words = [tokens[i.item()] for i in topk_indices]
            topk_scores = torch.gather(scores, 0, topk_indices)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": INSTRUCTION},
                          {"role": "user",  "content": PROMPT.format(tokens=topk_words)}],
                temperature=0.0
            )

            prediction = response.choices[0].message.content
            if prediction == data["target"]:
                n_true_predictions += 1
                plaus_scores.append(torch.sum(topk_scores).item())
            else:
                plaus_scores.append(0)

            # # compute metrics on Soft-NS and Soft-NC
            # print(f"Prompt: {data['prompt']}")
            # print(f"tokens: {topk_words}")
            # print(f"scores: {topk_scores}")
            # print(f'GPT prediction: {prediction}')
            # print("-"*10)
    print(f"true prediction rate: {n_true_predictions/len(samples)}")
    print(f"plausibility score: {torch.mean(torch.tensor(plaus_scores, dtype=torch.float16)).item()}")
    


if __name__ == "__main__":
    main()



