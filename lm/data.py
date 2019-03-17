import argparse
from pathlib import Path

import sentencepiece as spm
import tqdm


UNK = '<unk>'
END_OF_LINE = '<endofline>'
END_OF_TEXT = '<endoftext>'


def sp_train():
    parser = argparse.ArgumentParser(
        description='build sentencepiece model on train subset of the corpora')
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
        paths = []
        print(f'Reading corpora: {args.corpora}')
        for corpus_root in map(Path, args.corpora):
            train_root = corpus_root / 'train'
            corpus_paths = list(train_root.glob('**/*.txt'))
            if not corpus_paths:
                parser.error('Corpus train split {train_root} looks empty, '
                             'no text files found')
            paths.extend(corpus_paths)
        try:
            with sp_text.open('wt', encoding='utf8') as sp_text_file:
                for path in tqdm.tqdm(
                        paths, desc='building sentencepiece input'):
                    with path.open('rt', encoding='utf8') as f:
                        for line in f.readlines():
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
        f'--model_type=bpe',
        f'--max_sentence_length=16384',
        f'--bos_id=-1',
        f'--eos_id=-1',
        f'--unk_piece={UNK}',
        f'--control_symbols={END_OF_LINE},{END_OF_TEXT}',
        f'--character_coverage={args.character_coverage}',
    ]))


def encode_corpus():
    pass  # TODO


