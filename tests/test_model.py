import torch
import torch.cuda

from lm.model import HParams, Model, position_for, Norm, MLP
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


@parametrize_device('device')
def test_model(device):
    hparams = HParams(
        n_vocab=50,
        n_ctx=7,
        n_embed=16,
        n_head=4,
        n_layer=5,
    )
    model = Model(hparams)
    x = torch.LongTensor([[10, 1, 5, 45, 49, 0, 10],
                          [3,  6, 12, 8, 34, 9, 40]])
    x = x.to(device)
    model.to(device)
    result = model(x)
    print('result', result.shape)
    assert result.shape == (2, 7, 16)
