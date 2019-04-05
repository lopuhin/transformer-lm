"""
OpenAI's GPT-2 ported to PyTorch.
"""
import attr
import torch
from torch import nn


@attr.s(auto_attribs=True, frozen=True)
class HParams:
    n_vocab: int
    n_ctx: int
    n_embed: int
    n_head: int
    n_layer: int


class Model(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        self.hparams = hparams
        self.wpe = nn.Embedding(hparams.n_ctx, hparams.n_embed)
        self.wte = nn.Embedding(hparams.n_vocab, hparams.n_embed)

    def forward(self, x, past=None):
        results = {}
        past_length = 0 if past is None else past.shape[-2]
        batch_size, n_steps, *_ = x.shape
        position = position_for(batch_size, n_steps, past_length, x.device)
        h = self.wte(x) + self.wpe(position)
        assert h.shape == (batch_size, self.hparams.n_ctx, self.hparams.n_embed)
        return h


def position_for(batch_size, n_steps, past_length, device=None):
    return (torch.arange(past_length, n_steps + past_length, device=device)
            .unsqueeze(0).repeat(batch_size, 1))
