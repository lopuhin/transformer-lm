import torch
import torch.cuda

from lm.model import HParams, Model, position_for, Norm, MLP, Attention, Block
from .utils import parametrize_device


def test_position_for():
    assert torch.equal(position_for(batch_size=3, n_steps=5, past_length=2),
                       torch.LongTensor([[2, 3, 4, 5, 6]] * 3))


def test_norm():
    norm = Norm(10)
    x = torch.randn((2, 10)) * 10 + 5
    out = norm(x)
    assert out.shape == x.shape
    assert torch.allclose(out.mean(), torch.tensor(0.), atol=1e-6)
    assert torch.allclose(out.std(), torch.tensor(1.), atol=1e-1)


def test_mlp():
    x = torch.rand((2, 5, 16))
    mlp = MLP(16, 64)
    out = mlp(x)
    assert out.shape == x.shape


def test_attention_mask():
    assert torch.equal(
        Attention.attention_mask(5, 3, dtype=torch.float32),
        torch.tensor(
            [[0., 0., 0.],
             [0., 0., 0.],
             [1., 0., 0.],
             [1., 1., 0.],
             [1., 1., 1.]]))
    assert torch.equal(
        Attention.attention_mask(3, 5, dtype=torch.float32),
        torch.tensor(
            [[1., 1., 1., 0., 0.],
             [1., 1., 1., 1., 0.],
             [1., 1., 1., 1., 1.]]))


hparams = HParams(
    n_vocab=50,
    n_ctx=7,
    n_embed=32,
    n_hidden=32,
    n_head=4,
    n_layer=5,
    gradient_checkpointing=False,
)


def test_attention():
    attention = Attention(hparams)
    x = torch.randn(3, hparams.n_ctx, hparams.n_embed)
    a, present = attention(x, past=None)  # TODO test past
    assert a.shape == (3, hparams.n_ctx, hparams.n_embed)
    assert present.shape == (
        3, 2, hparams.n_head, hparams.n_ctx, hparams.n_embed // hparams.n_head)


def test_block():
    block = Block(hparams)
    x = torch.randn(3, hparams.n_ctx, hparams.n_embed)
    a, present = block(x, past=None)  # TODO test past
    assert a.shape == (3, hparams.n_ctx, hparams.n_embed)
    assert present.shape == (
        3, 2, hparams.n_head, hparams.n_ctx, hparams.n_embed // hparams.n_head)


@parametrize_device('device')
def test_model(device):
    model = Model(hparams)
    x = torch.LongTensor([[10, 1, 5, 45, 49, 0, 10],
                          [3,  6, 12, 8, 34, 9, 40]])
    x = x.to(device)
    model.to(device)
    result = model(x)
    assert result['logits'].shape == (2, hparams.n_ctx, hparams.n_vocab)
    assert result['presents'].shape == (
        2, hparams.n_layer, 2, hparams.n_head, hparams.n_ctx,
        hparams.n_embed // hparams.n_head)
