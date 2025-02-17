import logging

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseImportanceScoreEvaluator

from .utils import *
class NoiserImportanceScoreEvaluator(BaseImportanceScoreEvaluator):
    """Importance Score Evaluator
    
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, norm: str, mode: str) -> None:
        """Constructor

        Args:
            model: A Huggingface AutoModelForCausalLM model
            tokenizer: A Huggingface AutoTokenizer
            method: method

        """

        super().__init__(model, tokenizer)
        self.norm = norm
        self.mode = mode

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Evaluate importance score of input sequence

        Args:
            input_ids: input sequence [batch, sequence]
            target_id: target token [batch]

        Return:
            importance_score: evaluated importance score for each token in the input [batch, sequence]

        """

        input_text = [self.tokenizer.decode(i, skip_special_tokens=True) for i in input_ids]
        target_text = [self.tokenizer.decode(i) for i in torch.cat([input_ids, torch.unsqueeze(target_id, 0)], dim=1)]

        self.important_score = get_rationales(self.model,
                                              self.tokenizer,
                                              prompt=input_text,
                                              norm=self.norm,
                                              mode=self.mode,
                                              )
        return self.important_score

    @torch.no_grad()
    def rationalize(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Compute rational of a sequence on a target

        Args:
            input_ids: The sequence [batch, sequence]
            target_id: The target [batch]

        Return:
            pos_top_n: rational position in the sequence [batch, rational_size]

        """
        batch_importance_score = self.evaluate(input_ids, target_id)

        self.mean_important_score = torch.mean(batch_importance_score, dim=0)
        
        pos_sorted = torch.argsort(batch_importance_score, dim=-1, descending=True)

        return pos_sorted