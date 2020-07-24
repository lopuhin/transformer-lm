import json
from pathlib import Path
from typing import Dict

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

    def __len__(self):
        return len(self.vocab)

    @classmethod
    def load(cls, path: Path) -> 'CharTokenizer':
        return cls(json.loads(path.read_text('utf8')))
