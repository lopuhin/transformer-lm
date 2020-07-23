import json
from pathlib import Path
import re
from typing import List, Tuple, Optional, Dict

import sentencepiece as spm
import torch
import numpy as np

from .model import Model, HParams
from .common import END_OF_LINE, END_OF_TEXT, WORD_START


class ModelWrapper:
    END_OF_LINE = END_OF_LINE
    END_OF_TEXT = END_OF_TEXT

    def __init__(self, model: Model, sp_model: spm.SentencePieceProcessor,
                 params: Optional[Dict]):
        self.model = model
        self.sp_model = sp_model
        self.params = params or {}

    @classmethod
    def load(cls, root: Path):
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(root / 'sp.model'))
        params = json.loads((root / 'params.json').read_text())
        hparams = params['hparams']
        hparams.setdefault('n_hidden', hparams['n_embed'])
        pkl_path = root / 'model.pkl'
        if pkl_path.exists():
            model = torch.load(pkl_path, map_location='cpu')
        else:
            model = Model(HParams(**hparams))
            state = torch.load(root / 'model.pt', map_location='cpu')
            state_dict = fixed_state_dict(state['state_dict'])
            model.load_state_dict(state_dict)
            if 'seen_tokens' in state:
                params['seen_tokens'] = state['seen_tokens']
        return cls(model, sp_model, params=params)

    def tokenize(self, s: str) -> List[str]:
        return self.sp_model.EncodeAsPieces(s)

    def token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def id_to_token(self, token_id: int) -> str:
        return self.sp_model.IdToPiece(int(token_id))

    def get_log_probs(self, tokens: List[str]) -> torch.Tensor:
        """ Return a tensor with shape (len(tokens), len(self.sp_model)),
        with log-probabilities for tokens after each token in tokens.
        If this is a start of the text, you may want to prepend END_OF_TEXT:
        model.get_log_probs([model.END_OF_TEXT] + tokens).
        Use model.tokenize to obtain tokens.
        """
        return self._get_log_probs(tokens, past=None)[0]

    def _get_log_probs(
            self, tokens: List[str], past: Optional[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(tokens) > self.model.hparams.n_ctx:
            raise ValueError
        ids = [self.token_to_id(t) for t in tokens]
        ctx = torch.LongTensor(ids).unsqueeze(0)
        with torch.no_grad():
            output = self.model(ctx, past=past)
            logits = output['logits'].squeeze(0)
            return torch.log_softmax(logits, dim=1), output['presents']

    def get_occurred_log_probs(
            self, tokens: List[str]) -> List[Tuple[float, str]]:
        """ Return a list of log probs of actually occurred tokens,
        starting from the second.
        """
        log_probs = self.get_log_probs(tokens)
        out = []
        for idx, token in enumerate(tokens[1:]):
            out.append((float(log_probs[idx, self.token_to_id(token)]), token))
        return out

    def get_next_top_k(
            self, tokens: List[str], top_k: int,
            ) -> List[Tuple[float, str]]:
        """ Return a list of top k tuples of log prob and token,
        for what would come after the last token.
        """
        return self._get_next_top_k(tokens, top_k, past=None)[0]

    def _get_next_top_k(
            self, tokens: List[str], top_k: int,
            past: Optional[torch.Tensor],
            ) -> Tuple[List[Tuple[float, str]], torch.Tensor]:
        next_log_probs, presents = self._get_log_probs(tokens, past=past)
        next_log_probs = next_log_probs[-1]
        result = sorted(
            [(float(next_log_probs[i]), self.id_to_token(i))
             for i in next_log_probs.argsort()[-top_k:]],
            reverse=True)
        return result, presents

    def get_next_top_p_nucleus(
            self, tokens: List[str], top_p: float) -> torch.Tensor:
        """Return a list top top-p (by nucleus sampling) of log prob and token,
        for what would come after the last token.
        """
        return self._get_next_top_p_nucleus(tokens, top_p, past=None)[0]

    def _get_next_top_p_nucleus(
            self, tokens: List[str], top_p: float, past: Optional[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_log_probs, presents = self._get_log_probs(tokens, past=past)
        next_log_probs = next_log_probs[-1]
        sorted_log_probs, sorted_indices = \
            torch.sort(next_log_probs, descending=True)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_log_probs, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove)
        next_log_probs[indices_to_remove] = -float('Inf')
        return next_log_probs, presents

    def generate_tokens(
            self,
            tokens_prefix: List[str],
            tokens_to_generate: int,
            top_k: int,
            top_p: float = 0.0,
            temperature: float = 1.0,
            ) -> List[str]:
        tokens = list(tokens_prefix)
        output_tokens = []
        past = None

        for i in range(tokens_to_generate):

            if top_p <= 0.0:
                # generate TOP_K potential next tokens
                ntk, presents = self._get_next_top_k(tokens, top_k, past=past)

                # convert log probs to real probs
                logprobs = np.array(list(map(lambda a: a[0], ntk)))
                logprobs /= temperature
                probs = np.exp(logprobs) / np.exp(logprobs).sum()

                # pick next token randomly according to probs distribution
                next_token_n = np.random.choice(top_k, p=probs)
                next_token = ntk[next_token_n][1]
            else:
                filtered_logprobs, presents = self._get_next_top_p_nucleus(
                    tokens, top_p, past=past)
                filtered_logprobs /= temperature
                next_token_n = torch.multinomial(torch.nn.functional.softmax(
                    filtered_logprobs, dim=-1), num_samples=1)
                next_token = self.id_to_token(next_token_n)

            if past is None:
                past = presents
            else:
                past = torch.cat([past, presents], dim=-2)

            tokens = [next_token]
            output_tokens.append(next_token)

        return output_tokens

    def get_occurred_word_log_probs(
            self, tokens: List[str]) -> List[Tuple[float, str]]:
        """ Return a list of log probs of occurred words (not tokens!),
        starting from the second.
        """
        def is_word_start(token: str) -> bool:
            return token.startswith(WORD_START)

        def is_word(token: str) -> bool:
            return bool(re.search('\w', token))

        def is_continuation(token) -> bool:
            return is_word(token) and not is_word_start(token)

        def get_word(tokens: List[str]) -> str:
            return ''.join(t.lstrip(WORD_START) for t in tokens)

        log_probs = self.get_log_probs(tokens)
        output: List[Tuple[float, str]] = []
        all_tokens = [self.id_to_token(i) for i in range(len(self.sp_model))]
        continuation_mask = torch.tensor(list(map(is_continuation, all_tokens)))
        word_mask = torch.tensor(list(map(is_word, all_tokens)))
        current_word = []

        def finish_current_word(non_continuation_log_p=0):
            if not current_word:
                return
            word_log_p = sum(lp for lp, _ in current_word)
            output.append((
                float(word_log_p + non_continuation_log_p),
                get_word([t for _, t in current_word])))
            current_word.clear()

        for idx, token in enumerate(tokens[1:]):
            if not is_continuation(token):
                non_continuation_log_p = torch.logsumexp(
                    log_probs[idx, ~continuation_mask], dim=0)
                finish_current_word(non_continuation_log_p)
            if is_word(token):
                log_p = log_probs[idx, self.token_to_id(token)]
                words_log_p = torch.logsumexp(log_probs[idx, word_mask], dim=0)
                # discard punctuation probability as we score only words
                # (note that log_p is already included in words_log_p)
                log_p -= words_log_p
                current_word.append((log_p, token))
        # FIXME don't we need non_continuation_log_p here as well?
        finish_current_word()

        return output


def fixed_state_dict(state_dict):
    if all(k.startswith('module.') for k in state_dict):
        # legacy multi-GPU format
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict
