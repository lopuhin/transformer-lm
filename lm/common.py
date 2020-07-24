import json
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm


UNK = '<unk>'
END_OF_LINE = '<endofline>'
END_OF_TEXT = '<endoftext>'
WORD_START = '▁'


def load_tokenizer(path: Path):
    if path.name.endswith('.model'):
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(path))
    elif path.name.endswith('.json'):
        tokenizer = CharTokenizer.load(path)
    return tokenizer


def tokenizer_name(tokenizer):
    if isinstance(tokenizer, CharTokenizer):
        return 'chars.json'
    elif isinstance(tokenizer, spm.SentencePieceProcessor):
        return 'sp.model'
    else:
        raise ValueError(f'unexpected tokenizer type {type(tokenizer)}')


class CharTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.reverse_vocab = {i: c for c, i in self.vocab.items()}
        self.max_idx = max(self.reverse_vocab)
        self.unk_id = self.vocab[UNK]

    def __len__(self):
        return self.max_idx + 1

    @classmethod
    def load(cls, path: Path) -> 'CharTokenizer':
        return cls(json.loads(path.read_text('utf8')))

    def piece_to_id(self, x: str) -> int:
        assert len(x) == 1 or x in self.vocab
        return self.vocab.get(x, self.unk_id)

    def encode_as_ids(self, x: str) -> List[int]:
        return list(map(self.piece_to_id, x))

    def decode_ids(self, x: List[int]) -> str:
        return ''.join(self.reverse_vocab[i] for i in x)
