"""
OpenAI's GPT-2 ported to PyTorch.
"""
import math

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
        nn.init.normal_(self.wpe.weight, std=0.01)
        self.wte = nn.Embedding(hparams.n_vocab, hparams.n_embed)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(hparams) for _ in range(hparams.n_layer)])
        self.ln_f = Norm(self.hparams.n_embed)

    def forward(self, x, past=None):
        # Embedding
        past_length = 0 if past is None else past.shape[-2]
        batch_size, n_steps = x.shape
        position = position_for(batch_size, n_steps, past_length, x.device)
        h = self.wte(x) + self.wpe(position)
        assert h.shape == (batch_size, self.hparams.n_ctx, self.hparams.n_embed)
        return h  # TODO
        # Transformer
        presents = []
        for i, block in enumerate(self.blocks):
            h, present = block(h, past=past[:, i] if past is not None else None)
            presents.append(present)

        return {
            'presents': torch.stack(tuple(presents), dim=1),
            'logits': logits,
        }


class Block(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()

    def forward(self, x, past):
        pass


class Norm(nn.Module):
    """ Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    """
    def __init__(self, n_state, *, dim=-1, epsilon=1e-5):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))

    def forward(self, x):
        u = torch.mean(x, dim=self.dim, keepdim=True)
        xmu = x - u
        s = torch.mean(xmu * xmu, dim=self.dim, keepdim=True)
        return xmu * torch.rsqrt(s + self.epsilon) * self.g + self.b


class MLP(nn.Module):
    def __init__(self, n_features, n_state, w_init_std=0.02):
        super().__init__()
        self.c_fc = nn.Linear(n_features, n_state)
        nn.init.normal_(self.c_fc.weight, std=w_init_std)
        self.c_proj = nn.Linear(n_state, n_features)
        nn.init.normal_(self.c_proj.weight, std=w_init_std)

    def forward(self, x):
        x = gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x


def gelu(x, c=math.sqrt(2 / math.pi)):
    return 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3))))


def position_for(batch_size, n_steps, past_length, device=None):
    return (torch.arange(past_length, n_steps + past_length, device=device)
            .unsqueeze(0).repeat(batch_size, 1))
