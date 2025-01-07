from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):

    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
            self,
            log_probs: Tensor,
            log_probs_length: Tensor,
            text: List[str],
            **kwargs
    ) -> float:
        wers = []

        predictions = torch.argmax(log_probs.detach().cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().cpu().numpy()

        for pred_inds, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(pred_inds[:length])

            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers) if len(wers) > 0 else 0.0
