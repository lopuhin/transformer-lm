import torch

from lm.model import HParams, Model, position_for


def test_position_for():
    assert torch.equal(position_for(batch_size=3, n_steps=5, past_length=2),
                       torch.LongTensor([[2, 3, 4, 5, 6]] * 3))


# TODO test devices
def test_model():
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
    result = model(x)
    print('result', result.shape)
    assert result.shape == (2, 7, 16)
