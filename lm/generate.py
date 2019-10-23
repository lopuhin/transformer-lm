from pathlib import Path

import fire

from .fire_utils import only_allow_defined_args
from .inference import ModelWrapper


def gen_main(model_path, prefix, tokens_to_generate=42, top_k=8):
    print(f'loading model from {model_path}')
    mw = ModelWrapper.load(Path(model_path))

    print(f'generating text for prefix {prefix}')
    tokens = mw.tokenize(prefix)

    tokens_gen = mw.generate_tokens(tokens, tokens_to_generate, top_k)
    print(mw.sp_model.DecodePieces(tokens_gen))


def fire_gen_main():
    fire.Fire(only_allow_defined_args(gen_main))
