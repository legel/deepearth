# encoders/language/deepseek_v3_encoder.py
"""
Light wrapper that turns a DeepSeek checkpoint from Hugging Face
into a frozen encoder that emits hidden-state tokens with shape
(B, N, d_model).  A small Linear layer adapts whatever hidden size
the checkpoint uses to the shared `out_dim`.
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DeepSeekV3Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-llm-7b-base",
        out_dim: int = 1024,
        freeze: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone  = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device is None else {"": device},
        )

        if freeze:
            self.backbone.requires_grad_(False)

        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden_size, out_dim, bias=False)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args
        ----
        input_ids      : (B, N)
        attention_mask : (B, N)

        Returns
        -------
        hidden_states  : (B, N, out_dim)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        # outputs.last_hidden_state → (B, N, hidden_size)
        return self.proj(outputs.last_hidden_state)

