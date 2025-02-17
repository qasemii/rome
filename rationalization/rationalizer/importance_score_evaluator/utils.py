import torch
import random
import nltk

random.seed(42)
torch.set_grad_enabled(False)
nltk.download('punkt_tab')

import os
import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    Qwen2ForCausalLM,
    Gemma2ForCausalLM,
    LlamaForCausalLM,
)
from hf_olmo import OLMoForCausalLM

from .nethook import nethook

import nltk
nltk.download('punkt')

def make_noisy_embeddings(
    model,  # The model
    inp,  # A set of inputs
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    norm='inf',
    scale=1,  # Level of noise to add
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise

    if norm == '1':
        bound = lambda embed_dim: numpy.sqrt(2/numpy.pi) * embed_dim
    elif norm == '2':
        bound = lambda embed_dim: numpy.sqrt(embed_dim)
    elif norm == 'inf':
        bound = lambda embed_dim: numpy.sqrt(2 * numpy.log(embed_dim))
    elif norm == 'None':
        bound = lambda embed_dim: 1
    else:
        raise ValueError(f'Unknown norm: {norm}')

    prng = lambda *shape: rs.randn(*shape)/bound(shape[-1])
    noise_fn = lambda noise: scale * noise

    embed_layername = layername(model)

    def patch_rep(x):
        # If requested, we corrupt a range of token embeddings on batch items x[1:]
        b, e = tokens_to_mix

        ## Replace the target token embeddings with [MASK] embeddings ##########################################
        # tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        # mask_id = tokenizer.mask_token_id
        # mask_embedding = model.get_input_embeddings().weight[mask_id]
        # x[1:, b:e] = mask_embedding

        # Add noise to target token
        noise_data = noise_fn(torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))).to(x.device)
        x[1:, b:e] += noise_data
        return x

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername],
        edit_output=patch_rep,
    ):
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)
    return probs

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        device_map="auto",  # Use device map for efficient memory usage
        fp16=False,  # Use fp16 for reduced memory usage
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model_kwargs = {
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "torch_dtype": torch_dtype,
                "device_map": device_map,
            }
            if fp16:
                model_kwargs["torch_dtype"] = torch.float16  # Use fp16

            model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
            if isinstance(model, OLMoForCausalLM):
                model.tie_weights()

            model.eval()#.cuda()

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        self.tokenizer = tokenizer
        self.model = model

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

def layername(model):
    if isinstance(model, Qwen2ForCausalLM) or isinstance(model, Gemma2ForCausalLM) or isinstance(model, LlamaForCausalLM):
        return "model.embed_tokens"
    elif isinstance(model, OLMoForCausalLM):
        return "model.transformer.wte"
    else:
        raise ValueError(f"Unsupported model {type(self.model)}")

def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

def check_whitespace(prompt, tokens):
    results = []
    search_start = 0  # Track the current search position in the prompt

    for token in tokens:
        # Find the starting index of the token from the current position
        start_index = prompt.find(token, search_start)

        has_whitespace_before = start_index > 0 and prompt[start_index - 1].isspace()

        if has_whitespace_before:
            token = " " + token
            results.append(token)
        else:
            results.append(token)

        # Update position to search for the next token
        search_start = search_start + len(token)

    return results

def find_token_range(tokenizer, token_array, substring, start):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring, start)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)

def collect_token_range(tokenizer, prompt, tokens):
    inp = make_inputs(tokenizer, [prompt]) # NOTE: Mistral remove space before tokens !!!

    # Finding the range for each single token
    ranges = []
    start = 0
    for token in tokens:
        e_range = find_token_range(tokenizer, inp["input_ids"][0], token, start=start)
        ranges.append(e_range)
        start += len(token)  # Optimized this line by using '+=' instead of 'start = start + len(token)'

    return ranges

def predict_token(model, tokenizer, prompts, topk=None):
    inp = make_inputs(tokenizer, prompts)
    logits = model(**inp)["logits"]
    probs = torch.softmax(logits[:, -1, :], dim=-1)  # Correct slicing and dimension

    if topk:
        probs, preds = torch.topk(probs, topk, sorted=True, dim=-1)
        result = [
            (tokenizer.decode(pred.item()), prob.item())
            for pred, prob in zip(preds.squeeze(0), probs.squeeze(0))  # Squeeze batch dimension for single prompt
        ]

    else:
        probs, preds = torch.max(probs, dim=-1, keepdim=True)  # Keep dims for consistency
        result = (tokenizer.decode(preds.squeeze(0).item()), probs.squeeze(0).item())

    return result

def predict_from_input(model, inp):
    logits = model(**inp)["logits"]
    probs = torch.softmax(logits[:, -1, :], dim=1)

    return probs



def get_rationales(model, tokenizer, prompt, norm='inf', mode='prob'):
    # Use single prompt instead of 11
    device = model.device
    inp = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        logits = model(**inp)["logits"]
        probs = torch.softmax(logits[:, -1, :], dim=1)

    probs, preds = torch.max(probs, dim=-1, keepdim=True)  # Keep dims for consistency
    answer_id, base_score = (preds.squeeze(0).item(), probs.squeeze(0).item())


    tokens = nltk.word_tokenize(prompt)
    tokens = ['"' if token in ['``', "''"] else token for token in tokens]
    tokens = check_whitespace(prompt, tokens)
    tokens_range = collect_token_range(tokenizer, prompt, tokens)

    # Initialize on correct device
    tokens_score = torch.zeros(len(inp['input_ids'][0]), device=device)

    for idx, t_range in enumerate(tokens_range):
        b, e = t_range
        high = 1.0
        low = 0.0
        for _ in range(10):  # with 10 iteration the precision would be 2e-10 ~= 0.001
            k = (low + high) / 2
            with torch.no_grad():
                low_scores = make_noisy_embeddings(model, inp, norm=norm, tokens_to_mix=t_range, scale=k)
            prob = low_scores[answer_id].item()

            sorted_indices = torch.argsort(low_scores, descending=True)
            rank = (sorted_indices == answer_id).nonzero(as_tuple=True)[0].item()

            if rank == 0:
                low = k
            else:
                high = k

        if mode == 'noise':
            score = 1 - k
        elif mode == 'prob':
            score = base_score - prob
        else:
            raise ValueError(f'Invalid mode: {mode}')

        # Assign score to all subword tokens in the range
        tokens_score[b:e] = score

    tokens_score = tokens_score - torch.min(tokens_score)
    tokens_score = tokens_score / torch.sum(tokens_score)

    # Aggregate word scores by averaging sub-tokens scores
    word_scores = torch.tensor([
        tokens_score[b:e].sum().item() for b, e in tokens_range
    ], device=device)

    # Consider removing softmax if raw scores are preferred
    # word_scores = torch.softmax(word_scores, dim=-1)

    return {
        'input_tokens': tokens,
        'base_score': base_score,
        'word_scores': word_scores,
        'token_scores': tokens_score.unsqueeze(0)
    }
