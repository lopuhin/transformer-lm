import pytest
import torch.cuda


def parametrize_device(name):
    def deco(fn):
        return pytest.mark.parametrize([name], [
            ['cpu'],
            pytest.param(
                'cuda', marks=([pytest.mark.skip('cuda not available')]
                               if not torch.cuda.is_available() else [])
            )])(fn)
    return deco
