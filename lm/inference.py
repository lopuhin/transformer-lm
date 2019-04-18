import json
from pathlib import Path
from typing import List

import sentencepiece as spm
import torch

from .model import Model, HParams
from .data import END_OF_LINE, END_OF_TEXT


class ModelWrapper:
    END_OF_LINE = END_OF_LINE
    END_OF_TEXT = END_OF_TEXT

    def __init__(self, model: Model, sp_model: spm.SentencePieceProcessor):
        self.model = model
        self.sp_model = sp_model

    @classmethod
    def load(cls, root: Path):
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(root / 'sp.model'))
        hparams = json.loads((root / 'params.json').read_text())['hparams']
        hparams.setdefault('n_hidden', hparams['n_embed'])
        model = Model(HParams(**hparams))
        state = torch.load(root / 'model.pt', map_location='cpu')
        state_dict = fixed_state_dict(state['state_dict'])
        model.load_state_dict(state_dict)
        return cls(model, sp_model)

    def tokenize(self, s: str) -> List[str]:
        return self.sp_model.EncodeAsPieces(s)

    def token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def id_to_token(self, token_id: int) -> str:
        return self.sp_model.IdToPiece(int(token_id))

    def get_log_probs(self, tokens: List[str]):
        """ Return a tensor with shape (len(tokens), len(self.sp_model)),
        with log-probabilities for tokens after each token in tokens.
        If this is a start of the text, you may want to prepend END_OF_TEXT:
        model.get_log_probs([model.END_OF_TEXT] + tokens).
        Use model.tokenize to obtain tokens.
        """
        ids = [self.token_to_id(t) for t in tokens]
        ctx = torch.LongTensor(ids).unsqueeze(0)
        with torch.no_grad():
            # TODO - check prev argument for the model
            logits = self.model(ctx)['logits'].squeeze(0)
            return torch.log_softmax(logits, dim=1)

    def get_occurred_log_probs(self, tokens: List[str]):
        pass  # TODO


def fixed_state_dict(state_dict):
    if all(k.startswith('module.') for k in state_dict):
        # legacy multi-GPU format
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict
