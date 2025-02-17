import inseq
import torch

model_name = "meta-llama/Llama-3.1-8B"
# model_name = "google/gemma-2-2b"

model = inseq.load_model(model_name, "attention")
out = model.attribute(
  "To success",
)

attrs_list = [attr.aggregate().target_attributions[:-1] for attr in out.sequence_attributions]

attrs_batch = torch.permute(torch.cat(attrs_list, dim=1), dims=[1, 0])

print(attrs_batch)