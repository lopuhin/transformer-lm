import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
from typing import List

import numpy as np
import sentencepiece as spm
import tqdm

from .common import UNK, END_OF_TEXT, END_OF_LINE, load_tokenizer


def sp_train():
    parser = argparse.ArgumentParser(
        description='build sentencepiece model on train part of the corpora')
    arg = parser.add_argument
    arg('corpora', nargs='+',
        help='corpus roots, containing train/valid/test splits')
    arg('sp_text', help='text file for sentencepiece model '
                        '(will be used as-is if exists)')
    arg('sp_model_prefix', help='path (prefix) to output sentencepiece model')
    arg('--vocab-size', type=int, default=50000)
    arg('--character-coverage', type=float, default=1.0)
    args = parser.parse_args()

    sp_text = Path(args.sp_text)
    if sp_text.exists():
        print(f'Using existing "{sp_text}", remove and re-run if it is stale.')
    else:
        paths = _get_train_paths(args.corpora)
        try:
            with sp_text.open('wt', encoding='utf8') as sp_text_file:
                for path in tqdm.tqdm(
                        paths, desc='building sentencepiece input'):
                    with path.open('rt', encoding='utf8') as f:
                        for line in f:
                            if line.strip():
                                sp_text_file.write(line)
        except Exception:
            if sp_text.exists():
                sp_text.unlink()
            raise

    spm.SentencePieceTrainer.train(' '.join([
        f'--input={sp_text}',
        f'--model_prefix={args.sp_model_prefix}',
        f'--vocab_size={args.vocab_size}',
        '--model_type=bpe',
        '--max_sentence_length=16384',
        '--bos_id=-1',
        '--eos_id=-1',
        f'--unk_piece={UNK}',
        f'--control_symbols={END_OF_LINE},{END_OF_TEXT}',
        f'--character_coverage={args.character_coverage}',
    ]))


def char_train():
    parser = argparse.ArgumentParser(
        description='build char tokenizer on train subset of the corpora')
    arg = parser.add_argument
    arg('corpora', nargs='+',
        help='corpus roots, containing train/valid/test splits')
    arg('output', help='output path (.json) for the tokenizer')
    arg('--max-vocab-size', type=int, default=1000)
    args = parser.parse_args()

    paths = _get_train_paths(args.corpora)
    char_counts = Counter()
    for path in tqdm.tqdm(paths, desc='training tokenizer'):
        with path.open('rt', encoding='utf8') as f:
            for line in f:
                char_counts.update(line)
    vocab = {}
    char_counts.pop('\n')
    for ch, _ in char_counts.most_common(args.max_vocab_size):
        vocab[ch] = len(vocab)
    for ch in [END_OF_LINE, END_OF_TEXT, UNK]:
        vocab[ch] = len(vocab)
    Path(args.output).write_text(
        json.dumps(vocab, indent=4, ensure_ascii=False))


def _get_train_paths(corpora: List[str]) -> List[Path]:
    paths = []
    print(f'Reading corpora: {corpora}')
    for corpus_root in map(Path, corpora):
        train_root = corpus_root / 'train'
        corpus_paths = list(train_root.glob('**/*.txt'))
        if not corpus_paths:
            raise ValueError(
                f'Corpus train split {train_root} looks empty, '
                'no text files found')
        paths.extend(corpus_paths)
    return paths


def tokenize_corpus():
    # TODO support large corpora
    parser = argparse.ArgumentParser(
        description='encode corpus with a tokenizer')
    arg = parser.add_argument
    arg('corpora', nargs='+',
        help='corpus roots, containing train/valid/test splits')
    arg('tokenizer', help='path to tokenizer')
    arg('output', help='path to the output directory, '
                       'which will contain train.npy, valid.npy and test.npy')
    args = parser.parse_args()

    tokenizer = load_tokenizer(Path(args.tokenizer))
    eot = tokenizer.piece_to_id(END_OF_TEXT)
    eol = tokenizer.piece_to_id(END_OF_LINE)
    if len(tokenizer) < 2**8 - 1:
        dtype = np.uint8
    elif len(tokenizer) < 2**16 - 1:
        dtype = np.uint16
    else:
        dtype =np.uint32

    print(f'Reading corpora: {args.corpora}')
    encoded_splits = defaultdict(list)
    for corpus_root in map(Path, args.corpora):
        for split in ['train', 'valid', 'test']:
            split_root = corpus_root / split
            split_paths = list(split_root.glob('**/*.txt'))
            if not split_paths:
                parser.error(f'Corpus {split} split {split_root} looks empty, '
                             f'no text files found')

            def append_and_clear(x):
                encoded_splits[split].append(np.array(x, dtype=dtype))
                x.clear()

            for path in tqdm.tqdm(split_paths, desc=str(split_root)):
                encoded = []
                with path.open('rt', encoding='utf8') as f:
                    for line in f:
                        encoded.extend(tokenizer.encode_as_ids(line))
                        encoded.append(eol)
                        if len(encoded) > 100000:
                            # save memory by using a more compact representation
                            append_and_clear(encoded)
                    encoded.append(eot)
                append_and_clear(encoded)

    output_root = Path(args.output)
    output_root.mkdir(exist_ok=True, parents=True)
    for split in ['train', 'valid', 'test']:
        split_path = output_root / f'{split}.npy'
        print(f'Saving encoded split {split} to {split_path}')
        encoded = np.concatenate(encoded_splits[split])
        assert encoded.dtype == dtype
        np.save(split_path, encoded)
