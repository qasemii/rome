from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM, GPTJForCausalLM, Qwen2ForCausalLM
from .base import BaseEvaluator
from .sufficiency import SufficiencyEvaluator
from .comprehensiveness import ComprehensivenessEvaluator
from .soft_sufficiency import SoftSufficiencyEvaluator
from .soft_comprehensiveness import SoftComprehensivenessEvaluator
import logging

class SoftNormalizedSufficiencyEvaluator(BaseEvaluator):

    @override
    def __init__(self, model: AutoModelForCausalLM) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM
            rationale_ratio: ratio of rational tokens

        """
        super().__init__()
        self.model = model
        self.soft_sufficiency_evaluator_0 = SufficiencyEvaluator(model, rationale_ratio=0)
        #self.soft_sufficiency_evaluator_0 = SoftSufficiencyEvaluator(model)
        self.soft_sufficiency_evaluator = SoftSufficiencyEvaluator(model)

    @torch.no_grad()
    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_scores: torch.Tensor, input_wte: torch.Tensor = None, prob_original: torch.Tensor = None) -> torch.Tensor:
        """ Evaluate Comprehensiveness

        Args:
            input_ids: input token ids [batch, sequence]
            target_id: target token id [batch] (Deprecated)
            importance_scores: importance_scores of input tokens [batch, sequence]
            input_wte: input word token embedding (Optional)
            prob_original: probabilities of original input (Optional)

        Return:
            score [batch]

        """

        if input_wte == None:
            # input_wte = self.model.transformer.wte.weight[input_ids,:]
            if isinstance(self.model, GPT2LMHeadModel):
                gpt2Model: GPT2LMHeadModel = self.model
                input_wte = gpt2Model.transformer.wte.weight[input_ids,:]
            elif isinstance(self.model, OPTForCausalLM):
                optModel: OPTForCausalLM = self.model
                input_wte = optModel.model.decoder.embed_tokens(input_ids)
            elif isinstance(self.model, GPTJForCausalLM):
                gptjModel: GPTJForCausalLM = self.model
                input_wte = gptjModel.transformer.wte.weight[input_ids,:]
            elif isinstance(self.model, Qwen2ForCausalLM):
                qwen2Model: Qwen2ForCausalLM = self.model
                input_wte = qwen2Model.model.embed_tokens.weight[input_ids,:]
            else:
                raise ValueError(f"Unsupported model {type(self.model)}")
                # ValueError: Unsupported model <class 'transformers.models.gptj.modeling_gptj.GPTJForCausalLM'>

        # original prob
        if prob_original == None:
            logits_original = self.model(inputs_embeds=input_wte)["logits"]
            prob_original = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1)
        

        soft_sufficiency = self.soft_sufficiency_evaluator.evaluate(input_ids, None, importance_scores, input_wte, prob_original)
        logging.debug(' ')
        logging.debug(f"soft_sufficiency==>> {soft_sufficiency}")
        soft_sufficiency_0 = self.soft_sufficiency_evaluator_0.evaluate(input_ids, None, importance_scores, input_wte, prob_original)
        logging.debug(f"soft sufficiency_0==>> {soft_sufficiency_0}")
        #soft_norm_sufficiency = torch.clamp((soft_sufficiency - sufficiency_0), min=0, max=10) / (1 - sufficiency_0)
        #soft_norm_sufficiency = soft_sufficiency

        soft_norm_sufficiency = max(0, soft_sufficiency-soft_sufficiency_0)/(1-soft_sufficiency_0)


        return soft_norm_sufficiency
