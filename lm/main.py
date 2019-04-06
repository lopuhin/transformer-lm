import json
from pathlib import Path
import statistics
import shutil
import sys

import attr
import fire
import numpy as np
import torch.cuda
from torch import nn, optim
import tqdm
import sentencepiece as spm
import json_log_plots

from .fire_utils import only_allow_defined_args
from .model import Model, HParams


@only_allow_defined_args
def main(
        run_path,
        dataset_path,
        sp_model_path,
        epochs=10,
        lr=2.5e-4,
        batch_size=2,
        accum_gradients=1,  # accumulate gradients N times
        n_ctx=1024,
        n_embed=768,
        n_head=12,
        n_layer=12,
        clean=False,  # clean run folder
        log_every=1,
        save_every=1,
        ):
    run_path = Path(run_path)
    run_path_mark = run_path / '.lm'
    if clean and run_path.exists():
            assert run_path_mark.exists()  # to avoid removing unrelated folder
            shutil.rmtree(run_path)
    run_path.mkdir(exist_ok=True, parents=True)
    run_path_mark.touch()

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    hparams = HParams(
        n_vocab=len(sp_model),
        n_ctx=n_ctx,
        n_embed=n_embed,
        n_head=n_head,
        n_layer=n_layer,
    )
    params = dict(
        hparams=attr.asdict(hparams),
        argv=' '.join(sys.argv),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        accum_gradients=accum_gradients,
    )
    params_s = json.dumps(params, indent=4, sort_keys=True, ensure_ascii=False)
    print(params_s)
    (run_path / 'params.json').write_text(params_s, encoding='utf8')

    dataset_path = Path(dataset_path)
    print(f'Loading dataset from {dataset_path}')
    valid_dataset = np.load(dataset_path / 'valid.npy')
    train_dataset = np.load(dataset_path / 'train.npy')
    print(f'Train dataset has {len(train_dataset):,} tokens')
    print(f'Validation dataset has {len(valid_dataset):,} tokens')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(hparams).to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_step():
        context = _gen_batch(
            train_dataset, n_ctx=n_ctx, batch_size=batch_size * accum_gradients)
        context = torch.LongTensor(context, device=device)
        optimizer.zero_grad()
        loss_value = 0.
        for ctx in torch.split(context, batch_size):
            logits = model(ctx)['logits']
            loss = loss_fn(input=logits[:, :-1].reshape([-1, logits.shape[-1]]),
                           target=ctx[:, 1:].reshape(-1))
            loss.backward()
            loss_value += float(loss.item()) / accum_gradients
        optimizer.step()
        return loss_value

    def save():
        torch.save({
            'state_dict': model.state_dict(),
            'step': step,
        }, run_path / 'model.pt')

    step = 1
    step_tokens = n_ctx * batch_size * accum_gradients
    epoch_size = len(train_dataset) // step_tokens
    loss_meter = AverageMeter()
    try:
        for epoch in tqdm.trange(1, epochs + 1, desc='epoch',
                                 dynamic_ncols=True):
            epoch_pbar = tqdm.trange(epoch_size, desc=f'epoch {epoch}',
                                     dynamic_ncols=True)
            for _ in epoch_pbar:
                if step % save_every == 0:
                    save()
                lv = train_step()
                loss_meter.update(lv)
                step += 1
                epoch_pbar.set_postfix({
                    'step': step,
                    'loss': f'{loss_meter.mean():.2f}'})
                if step % log_every == 0:
                    json_log_plots.write_event(
                        run_path,
                        step=step * step_tokens,
                        loss=loss_meter.mean())
                    loss_meter.reset()

    except KeyboardInterrupt:
        print('Interrupted, saving')
        save()
        sys.exit(1)


def _gen_batch(dataset: np.ndarray, n_ctx: int, batch_size: int):
    indices = [np.random.randint(0, len(dataset) - n_ctx)
               for _ in range(batch_size)]
    return [dataset[idx: idx + n_ctx] for idx in indices]


class AverageMeter:
    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(value)

    def mean(self):
        return statistics.mean(self.values)

    def reset(self):
        self.values.clear()


def fire_main():
    fire.Fire(main)
