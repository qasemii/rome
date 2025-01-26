import argparse
import json
import logging
import os
from tqdm import tqdm

import torch
from dsets.data_utils import create_analogy_templates, preprocess_analogies
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--analogies-file", type=str, default="data/analogies.txt", help="")  # TODO
parser.add_argument("--output-dir", type=str, default="data/analogies", help="")  # TODO
parser.add_argument("--compact-output", type=bool, default=True, help="")  # TODO
parser.add_argument("--schema-uri", type=str, default="../../docs/analogy.schema.json", help="")  # TODO
parser.add_argument("--device", type=str, default="cuda", help="")  # TODO

parser.add_argument("--model", type=str, default="gpt2-medium", help="")  # TODO # gpt2-medium gpt2-large
parser.add_argument("--cache_dir", type=str, default=None, help="store models")
args = parser.parse_args()

analogies_file = args.analogies_file
output_dir = args.output_dir
compact_output = args.compact_output
schema_uri = args.schema_uri
device = args.device

# Load analogies

with open(analogies_file) as f:
    analogies = f.readlines()
analogies = [line.rstrip("\n") for line in analogies]
# print("00".center(50, "-"))
# print(f"==>> analogies: {analogies}")

# Load model

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)
model.to(device)

with torch.no_grad():
    # Build analogies index
    all_analogies = preprocess_analogies(analogies, tokenizer)
    all_analogies = create_analogy_templates(all_analogies)

    # Build data
    data = list()
    data_id = 0
    for analogy_idx, (analogy_label, analogy_config) in tqdm(enumerate(all_analogies.items())):
        analogy_config = all_analogies[analogy_label]

        template = analogy_config["template"]

        # map tags to target/relative
        target_tag = "a" if template.index("[A]") > template.index("[B]") else "b"
        relative_tag = "a" if target_tag == "b" else "b"

        for pair_idx in range(len(analogy_config["a"])):
            word_a = analogy_config["a"][pair_idx] # target word
            word_b = analogy_config["b"][pair_idx] # relative word

            prompt = template.replace(" [A]", "").replace("[B]", word_b)

            distractor_start_id = prompt.index("(")-1
            distractor_end_id = prompt.index(")")+2
            distractor = prompt[distractor_start_id:distractor_end_id]

            distractor_removed = prompt.replace(distractor, "")

            data.append({
                "id": data_id,
                "prompt": prompt,
                "target": word_a,
                "relative": word_b,
            })
            data_id += 1

    json_str = json.dumps(data)
    filename = os.path.join(output_dir, "LongRA.json")
    with open(filename, "w") as f_output:
        f_output.write(json_str)

    logging.info(f"Preparation finished")