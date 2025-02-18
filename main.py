import argparse
import os
import torch
import random
import nltk
import pickle
from tqdm import tqdm
from pathlib import Path
from util.globals import DATA_DIR

from dsets import (
    KnownsDataset,
    CounterFactDataset,
)

from rationalization.src.evaluation.evaluator.soft_norm_sufficiency import SoftNormalizedSufficiencyEvaluator
from rationalization.src.evaluation.evaluator.soft_norm_comprehensiveness import SoftNormalizedComprehensivenessEvaluator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

device = "cuda"

random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_grad_enabled(False)


def predict_token(model, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors='pt').to(device)
    logits = model(**inp)["logits"]
    probs = torch.softmax(logits[:, -1, :], dim=-1) 
    probs, preds = torch.max(probs, dim=-1, keepdim=True)  # Keep dims for consistency
    result = tokenizer.decode(preds.squeeze(0).item())
    return result


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
    aa("--method", default="noiser", type=str,
       help="noiser, attention, attention_last, attention_rollout, gradient_shap,\
             input_x_gradient, integrated_gradients, lime, reagent, occlusion")  

    args = parser.parse_args()

    result_dir = f"{args.output_dir}{args.dataset}/{args.model_name}"
    os.makedirs(result_dir, exist_ok=True)

    cache_dir = f"cache/{args.model_name}"
    os.makedirs(cache_dir, exist_ok=True)

    print('Loading model and tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

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
        d for d in dataset
        if predict_token(model, tokenizer, d['prompt']).strip() == d['target']
    ]
    print(f"Filtered dataset to {len(dataset)} examples")

    if args.method == 'noiser':
        nltk.download('punkt_tab')
        from rationalization.rationalizer.importance_score_evaluator.noiser import  NoiserImportanceScoreEvaluator
        rationalizer = NoiserImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            norm=args.norm,
            mode=args.mode
        )
    elif args.method == 'random':
        pass
    elif args.method == 'attention_last' or args.method == 'attention_rollout':
        from rationalization.rationalizer.importance_score_evaluator.attention import AttentionImportanceScoreEvaluator
        rationalizer = AttentionImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            attn_type=args.method.replace("attention_", "")
        )
    else: #['integrated_gradients', 'input_x_gradient', 'attention', 'gradient_shap']
        # assert args.method in ['integrated_gradients', 'input_x_gradient', 'attention', 'gradient_shap'] # input_x_gradient = signed in self written
        from rationalization.rationalizer.importance_score_evaluator.inseq import  InseqImportanceScoreEvaluator
        rationalizer = InseqImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            method=args.method, 
            attribute_params={}
        )

    # init evaluator
    soft_norm_suff_evaluator = SoftNormalizedSufficiencyEvaluator(model)
    soft_norm_comp_evaluator = SoftNormalizedComprehensivenessEvaluator(model)

    source_soft_ns = []
    source_soft_nc = []
    random_soft_ns = []
    random_soft_nc = []

    print("Starting rationalization ...")
    samples = dataset if args.n_samples == -1 else random.choices(dataset, k=args.n_samples)
    for data in tqdm(samples):
        idx = data['id']

        input_ids = tokenizer(data["prompt"], return_tensors='pt')['input_ids'][0].to(device)
        attention_mask = tokenizer(data["prompt"], return_tensors='pt')['attention_mask'][0].to(device)
        generated_ids = model.generate(input_ids=torch.unsqueeze(input_ids, 0),
                                       attention_mask= torch.unsqueeze(attention_mask, 0),
                                       pad_token_id=tokenizer.pad_token_id,
                                       max_new_tokens=args.max_new_tokens, 
                                       do_sample=False)[0]
        
        target_id = tokenizer(" "+data["target"], return_tensors='pt')['input_ids'][0].to(device)
        # generated_texts = tokenizer.decode(generated_ids)
        # print(f'generated full sequence --> {generated_texts}')

        s_soft_ns = []
        s_soft_nc = []
        r_soft_ns = []
        r_soft_nc = []

        score_map = torch.zeros([generated_ids.shape[0] - input_ids.shape[0], generated_ids.shape[0] - 1], device=device)
        for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0]):
            input_ids = torch.unsqueeze(generated_ids[:target_pos], 0)
            target_id = torch.unsqueeze(generated_ids[target_pos], 0)

            # rationalization
            rationalizer.rationalize(input_ids, target_id)
            scores = rationalizer.mean_important_score.unsqueeze(0).to(device)
            # if args.method=='occlusion':
            #     scores = scores/torch.sum(scores)

            random_score = torch.softmax(torch.rand(scores.shape, device=device), dim=-1)

            # compute Soft-NS and Soft-NC on source importance score
            s_soft_ns_step = soft_norm_suff_evaluator.evaluate(input_ids, target_id, scores)
            s_soft_ns.append(s_soft_ns_step)

            s_soft_nc_step = soft_norm_comp_evaluator.evaluate(input_ids, target_id, scores)
            s_soft_nc.append(s_soft_nc_step)

            # compute Soft-NS and Soft-NC on random importance score
            r_soft_ns_step = soft_norm_suff_evaluator.evaluate(input_ids, target_id, random_score)
            r_soft_ns.append(r_soft_ns_step)

            r_soft_nc_step = soft_norm_comp_evaluator.evaluate(input_ids, target_id, random_score)
            r_soft_nc.append(r_soft_nc_step)

        # # compute metrics on Soft-NS and Soft-NC
        # soft_ns = torch.log(torch.sum(torch.tensor(source_soft_ns)) / torch.sum(torch.tensor(random_soft_ns)))
        # n_soft_ns.append(soft_ns.item())

        # soft_nc = torch.log(torch.sum(torch.tensor(source_soft_nc)) / torch.sum(torch.tensor(random_soft_nc)))
        # n_soft_nc.append(soft_nc.item())

        source_soft_ns.append(torch.mean(torch.tensor(s_soft_ns)).item())
        source_soft_nc.append(torch.mean(torch.tensor(s_soft_nc)).item())

        random_soft_ns.append(torch.mean(torch.tensor(r_soft_ns)).item())
        random_soft_nc.append(torch.mean(torch.tensor(r_soft_nc)).item())

    soft_ns = torch.log(torch.mean(torch.tensor(source_soft_ns)) / torch.mean(torch.tensor(random_soft_ns)))
    soft_nc = torch.log(torch.mean(torch.tensor(source_soft_nc)) / torch.mean(torch.tensor(random_soft_nc)))

    print(f"Normalized Soft-NS: {soft_ns}")
    print(f"Normalized Soft-NC: {soft_nc}")
    
    # export results
    # Path(result_dir).mkdir(exist_ok=True, parents=True)
    # method = f'{args.method}_{args.norm}_{args.mode}' if args.method == 'noiser' else args.method
    # with open(os.path.join(result_dir, f'{method}.pkl'), 'wb') as outfile:
    #     pickle.dump(results, outfile)


if __name__ == "__main__":
    main()



