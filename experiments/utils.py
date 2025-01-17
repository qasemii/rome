import argparse
import json
import os
import re
from collections import defaultdict

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
    OlmoForCausalLM,
)
from hf_olmo import OLMoForCausalLM
from peft import AutoPeftModelForCausalLM

from dsets import KnownsDataset
from dsets.data_utils import check_whitespace

from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.globals import DATA_DIR
from util.runningstats import Covariance, tally

import nltk
nltk.download('punkt')

def make_noisy_embeddings(
    mt,  # The model
    inp,  # A set of inputs
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
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
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    embed_layername = layername(mt.model, 0, "embed")

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x):
        # If requested, we corrupt a range of token embeddings on batch items x[1:]
        if tokens_to_mix is not None:
            b, e = tokens_to_mix
            noise_data = noise_fn(
                torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
            ).to(x.device)
            if replace:
                x[1:, b:e] = mt.tokenizer.encode('[MASK]')
            else:
                x[1:, b:e] += noise_data
        return x

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        mt.model,
        [embed_layername],
        edit_output=patch_rep,
    ) as td:
        outputs_exp = mt.model(**inp)

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

        self.tokenizer = tokenizer
        self.model = model

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

def layername(model, num, kind=None):
    if isinstance(model, Qwen2ForCausalLM):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    elif isinstance(model, OLMoForCausalLM):
        if kind == "embed":
            return "model.transformer.wte"
        elif kind in ['attn_out', 'ff_out', 'att_proj', 'ff_proj', 'mlp']:
            return f'model.transformer.block.{num}{"" if kind is None else "." + kind}'
        else:
            assert False, "Please choose one of the following: ['attn_out', 'ff_out', 'att_proj', 'ff_proj']"
    elif isinstance(model, Gemma2ForCausalLM):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    elif isinstance(model, LlamaForCausalLM):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
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
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

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

def predict_token(mt, prompts, topk=None):
    inp = make_inputs(mt.tokenizer, prompts)
    logits = mt.model(**inp)["logits"]
    probs = torch.softmax(logits[:, -1, :], dim=-1)  # Correct slicing and dimension

    if topk:
        probs, preds = torch.topk(probs, topk, sorted=True, dim=-1)
        result = [
            (mt.tokenizer.decode(pred.item()), prob.item())
            for pred, prob in zip(preds.squeeze(0), probs.squeeze(0))  # Squeeze batch dimension for single prompt
        ]

    else:
        probs, preds = torch.max(probs, dim=-1, keepdim=True)  # Keep dims for consistency
        result = (mt.tokenizer.decode(preds.squeeze(0).item()), probs.squeeze(0).item())

    return result

def predict_from_input(model, inp):
    logits = model(**inp)["logits"]
    probs = torch.softmax(logits[:, -1, :], dim=1)

    return probs

def collect_embedding_std(mt, subjects):
    alldata = []
    for s in tqdm(subjects):
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level

def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100

    progress = lambda x, **k: x

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                del batch["position_ids"]
                with nethook.Trace(model, layername(mt.model, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()

def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer

def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)

def collect_embedding_tdist(mt, degree=3):
    # We will sample sqrt(degree / u) * sample, where u is from the chi2[degree] dist.
    # And this will give us variance is (degree / degree - 2) * cov.
    # Therefore if we want to match the sample variance, we should
    # reduce cov by a factor of (degree - 2) / degree.
    # In other words we should be sampling sqrt(degree - 2 / u) * sample.
    u_sample = torch.from_numpy(
        numpy.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        student = factor * gauss
        return student

    return normal_to_student

def collect_token_range(mt, prompt, window=1):
    inp = make_inputs(mt.tokenizer, [prompt])

    # Tokenize sentence into words and punctuation
    tokens = nltk.word_tokenize(prompt)
    tokens = ['"' if token in ['``', "''"] else token for token in tokens]
    tokens = check_whitespace(prompt, tokens)

    # Finding the range for each single token
    ranges = []
    start = 0
    for token in tokens:
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], token, start=start)
        ranges.append(e_range)
        start += len(token)  # Optimized this line by using '+=' instead of 'start = start + len(token)'

    # Computing the range based on the window size
    ranges = [(ranges[i][0], ranges[i+window-1][1]) for i in range(len(tokens) - window + 1)]

    return ranges