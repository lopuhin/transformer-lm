import argparse
from pathlib import Path

import sentencepiece as spm
import tqdm


END_OF_SENTENCE = '</s>'
END_OF_TEXT = '</text>'


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
    args = parser.parse_args()

    sp_text = Path(args.sp_text)
    if not sp_text.exists():
        paths = []
        for corpus_root in map(Path, args.corpus):
            train_root = corpus_root / 'train'
            paths = list(train_root.glob('**/*.txt'))
            if not paths:
                parser.error('Corpus train split {train_root} looks empty, '
                             'no text files found')
        try:
            with sp_text.open('wt', encoding='utf8') as sp_text_file:
                for path in tqdm.tqdm(
                        paths, desc='building sentencepiece input'):
                    with path.open('rt', encoding='utf8') as f:
                        for line in f.readlines():
                            if line != '\n':
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
        f'--bos_id=-1',
        f'--max_sentence_length=16384',
        f'--eos_piece={END_OF_SENTENCE}',
        f'--control_symbols={END_OF_TEXT}',
    ]))


def encode_corpus():
    pass  # TODO


