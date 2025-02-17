import argparse
import os
import torch
import random
import nltk
import pickle
from tqdm import tqdm
from pathlib import Path
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

from rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
from rationalization.rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from rationalization.rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from rationalization.rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from rationalization.rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler

from rationalization.src.evaluation.evaluator.soft_norm_sufficiency import SoftNormalizedSufficiencyEvaluator
from rationalization.src.evaluation.evaluator.soft_norm_comprehensiveness import \
    SoftNormalizedComprehensivenessEvaluator

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
       help="noiser, attention, attention_last, attention_rollout, \
             gradient_shap, input_x_gradient, integrated_gradients, lime, reagent, occlusion")  # TODO

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

    special_tokens_dict = dict()
    if mt.tokenizer.pad_token is None:
        mt.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        special_tokens_dict["pad_token"] = mt.tokenizer.eos_token
    if mt.tokenizer.bos_token is None:
        mt.tokenizer.add_special_tokens({'bos_token': '<bos>'})
        special_tokens_dict["bos_token"] = mt.tokenizer.bos_token
    if mt.tokenizer.eos_token is None:
        mt.tokenizer.add_special_tokens({'eos_token': '<eos>'})
        special_tokens_dict["eos_token"] = mt.tokenizer.eos_token
    if mt.tokenizer.unk_token is None:
        mt.tokenizer.add_special_tokens({'unk_token': '<unk>'})
        special_tokens_dict["unk_token"] = mt.tokenizer.unk_token
    # if mt.tokenizer.bos_token is None:
    #    mt.tokenizer.bos_token = "<bos>"
    #    mt.tokenizer.bos_token_id = mt.tokenizer.convert_tokens_to_ids("<bos>")
    
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
    elif args.method == 'attention_last' or args.method == 'attention_rollout':
        from rationalization.rationalizer.importance_score_evaluator.attention import AttentionImportanceScoreEvaluator
        rationalizer = AttentionImportanceScoreEvaluator(
            model=mt.model,
            tokenizer=mt.tokenizer,
            attn_type=args.method.replace("attention_", "")
        )
    else:
        # assert args.method in ['integrated_gradients', 'input_x_gradient', 'attention', 'gradient_shap'] # input_x_gradient = signed in self written
        from rationalization.rationalizer.importance_score_evaluator.inseq import  InseqImportanceScoreEvaluator
        rationalizer = InseqImportanceScoreEvaluator(
            model=mt.model,
            tokenizer=mt.tokenizer,
            method=args.method, 
            attribute_params=special_tokens_dict
        )

    # init evaluator
    soft_norm_suff_evaluator = SoftNormalizedSufficiencyEvaluator(mt.model)
    soft_norm_comp_evaluator = SoftNormalizedComprehensivenessEvaluator(mt.model)

    source_soft_ns = []
    source_soft_nc = []
    random_soft_ns = []
    random_soft_nc = []
    results = []
    print("Starting rationalization ...")

    samples = dataset if args.n_samples == -1 else random.choices(dataset, k=args.n_samples)
    for data in tqdm(samples):
        idx = data['id']

        input_ids = mt.tokenizer(data["prompt"], return_tensors='pt')['input_ids'][0].to(mt.model.device)
        target_id = mt.tokenizer(" "+data["target"], return_tensors='pt')['input_ids'][0].to(mt.model.device)
        # generated_texts = mt.tokenizer.decode(generated_ids)
        # print(f'generated full sequence --> {generated_texts}')

        # Gemma and Llama add [bos] token which should be exclude from input prompt when
        if args.method == 'noiser':
            ers = get_rationales(mt,
                                 data["prompt"],
                                 norm=args.norm,
                                 mode=args.mode
                                 )
            scores = ers['token_scores']
        elif args.method == 'random':
            scores = torch.softmax(
                torch.rand(torch.unsqueeze(input_ids, 0).shape, device=mt.model.device), dim=-1)
        else:
            rationalizer.rationalize(torch.unsqueeze(input_ids, 0), torch.unsqueeze(target_id, 0))
            scores = rationalizer.mean_important_score.unsqueeze(dim=0).to(mt.model.device)
            if args.method=='occlusion':
                scores = scores/torch.sum(scores)
        # importance score by Random Score
        rand_scores = torch.softmax(
            torch.rand(torch.unsqueeze(input_ids, 0).shape, device=mt.model.device), dim=-1)

        try:
            # compute Soft-NS and Soft-NC on source importance score
            source_soft_ns_step = soft_norm_suff_evaluator.evaluate(torch.unsqueeze(input_ids, 0),
                                                                    torch.unsqueeze(target_id, 0), scores)
            source_soft_nc_step = soft_norm_comp_evaluator.evaluate(torch.unsqueeze(input_ids, 0),
                                                                    torch.unsqueeze(target_id, 0), scores)
            # print(f"Source Soft-NS: {source_soft_ns_step}, Source Soft-NC: {source_soft_nc_step}")

            # compute Soft-NS and Soft-NC on random importance score
            random_soft_ns_step = soft_norm_suff_evaluator.evaluate(torch.unsqueeze(input_ids, 0),
                                                                    torch.unsqueeze(target_id, 0), rand_scores)
            random_soft_nc_step = soft_norm_comp_evaluator.evaluate(torch.unsqueeze(input_ids, 0),
                                                                    torch.unsqueeze(target_id, 0), rand_scores)
            # print(f"Random Soft-NS: {random_soft_ns_step}, Random Soft-NC: {random_soft_nc_step}")

            source_soft_ns.append(source_soft_ns_step.item())
            source_soft_nc.append(source_soft_nc_step.item())

            random_soft_ns.append(random_soft_ns_step.item())
            random_soft_nc.append(random_soft_nc_step.item())

        except:
            print(f"Unable to get the score for {idx}")
            continue

        results.append({'id': data["id"],
                        'prompt': data["prompt"],
                        'target': data["target"],
                        'attributions': scores})

            
    # # compute metrics on Soft-NS and Soft-NC
    metric_soft_ns = torch.log(torch.mean(torch.tensor(source_soft_ns)) / torch.mean(torch.tensor(random_soft_ns)))
    metric_soft_nc = torch.log(torch.mean(torch.tensor(source_soft_nc)) / torch.mean(torch.tensor(random_soft_nc)))
    print(f"metric_soft_ns: {metric_soft_ns}, metric_soft_nc: {metric_soft_nc}")
    

    # export results
    # Path(result_dir).mkdir(exist_ok=True, parents=True)
    # method = f'{args.method}_{args.norm}_{args.mode}' if args.method == 'noiser' else args.method
    # with open(os.path.join(result_dir, f'{method}.pkl'), 'wb') as outfile:
    #     pickle.dump(results, outfile)


if __name__ == "__main__":
    main()



