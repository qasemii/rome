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

from experiments.utils import check_whitespace
from dsets.data_utils import match_tokens_with_scores
from experiments.utils import collect_token_range

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


def rationalize(prompt="Hello", 
                model_name="Qwen/Qwen2-0.5B", 
                method="noiser",
                max_new_tokens=1, 
                norm="2", 
                mode="prob"):
    
    nltk.download('averaged_perceptron_tagger_eng')

    print('Loading model and tokenizer ...')
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    pad_token_id = mt.tokenizer.pad_token_id if mt.tokenizer.pad_token_id is not None else mt.tokenizer.eos_token_id
    # init rationalizer
    rational_size = 3
    rational_size_ratio = None

    # tested with 3 0.1 5000 5
    stopping_top_k = 3
    replacing = 0.1
    max_step = 3000
    batch = 3

    if method == 'noiser':
        nltk.download('punkt_tab')
    elif method == 'random':
        pass
    elif method == 'reagent':

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
    elif method == 'attention_last' or method == 'attention_rollout':
        from rationalization.rationalizer.importance_score_evaluator.attention import \
            AttentionImportanceScoreEvaluator
        importance_score_evaluator = AttentionImportanceScoreEvaluator(
            model=mt.model,
            tokenizer=mt.tokenizer,
            attn_type=method.replace("attention_", "")
        )
        from rationalization.rationalizer.sample_rationalizer import SampleRationalizer
        rationalizer = SampleRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            top_n=3,
        )
    else:
        # assert method in ['integrated_gradients', 'input_x_gradient', 'attention', 'gradient_shap'] # input_x_gradient = signed in self written
        from rationalization.rationalizer.importance_score_evaluator.inseq import \
            InseqImportanceScoreEvaluator
        importance_score_evaluator = InseqImportanceScoreEvaluator(
            model=mt.model,
            tokenizer=mt.tokenizer,
            method=method,  # integrated_gradients input_x_gradient attention
            attribute_params={
            }
        )
        from rationalization.rationalizer.sample_rationalizer import SampleRationalizer
        rationalizer = SampleRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            top_n=3,
        )

    print("Starting rationalization ...")

    prompt = prompt
    input_ids = mt.tokenizer(prompt, return_tensors='pt')['input_ids'][0].to(mt.model.device)
    attention_mask = mt.tokenizer(prompt, return_tensors='pt')['attention_mask'][0].to(mt.model.device)

    generated_ids = mt.model.generate(input_ids=torch.unsqueeze(input_ids, 0),
                                        attention_mask=torch.unsqueeze(attention_mask, 0),
                                        max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        pad_token_id=pad_token_id)[0]
    full_text = mt.tokenizer.decode(generated_ids)
    # print(f'generated full sequence --> {generated_texts}')

    tokens = nltk.word_tokenize(full_text)
    tokens = ['"' if token in ['``', "''"] else token for token in tokens]
    tokens = check_whitespace(full_text, tokens)
    tokens_range = collect_token_range(mt, full_text, tokens)

    results = {'prompt': prompt,
                'target': mt.tokenizer.decode(generated_ids[input_ids.shape[0]:]),
                'entities': tokens,
                'attributions': {}
                }

    # Gemma and Llama add [bos] token which should be exclude from input prompt when
    start_pos = 1 if isinstance(mt.model, Gemma2ForCausalLM) or isinstance(mt.model, LlamaForCausalLM) else 0
    for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0]):
        target_id = generated_ids[target_pos]

        if method == 'noiser':
            prompt = mt.tokenizer.decode(generated_ids[:target_pos])
            ers = get_rationales(mt,
                                prompt,
                                norm=norm,
                                mode=mode,)
            scores = ers['word_scores']
        elif method == 'random':
            scores = torch.softmax(
                torch.rand((len(tokens)), device=mt.model.device), dim=-1)
        else:
            rationalizer.rationalize(torch.unsqueeze(generated_ids[:target_pos], 0), torch.unsqueeze(target_id, 0))
            scores = rationalizer.mean_important_score.unsqueeze(dim=0).to(mt.model.device)
            scores = match_tokens_with_scores(scores.squeeze(), tokens_range)
            

        results['attributions'][tokens[target_pos]] = scores

    print(results)
    return results



