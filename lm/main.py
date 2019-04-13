import json
import os
from pathlib import Path
import statistics
import shutil
import sys

import attr
import fire
import json_log_plots
import numpy as np
import torch.cuda
import torch.distributed
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch import nn, optim
import tqdm
import sentencepiece as spm

from .fire_utils import only_allow_defined_args, get_defined_args
from .model import Model, HParams


def main(
        run_path,
        dataset_path,
        sp_model_path,
        epochs=10,
        lr=2.5e-4,
        batch_size=2,
        accum_gradients=32,  # accumulate gradients N times
        n_ctx=1024,
        n_embed=768,
        n_head=12,
        n_layer=12,
        clean=False,  # clean run folder
        log_every=1,
        save_every=1000,
        max_steps=None,
        master_port='40390',
        master_addr='127.0.0.1',
        # These are set automatically when multiple GPUs are available
        device_id=None,
        n_devices=None,
        ):
    if n_devices is None:
        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            locals_ = locals()
            kwargs = {a: locals_[a] for a in get_defined_args(main)}
            mp.spawn(_main_mp, (kwargs,), n_devices)
            return

    is_main = device_id in {0, None}
    world_size = max(1, n_devices)

    run_path = Path(run_path)
    if is_main:
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
    if is_main:
        print(params_s)
        (run_path / 'params.json').write_text(params_s, encoding='utf8')

    dataset_path = Path(dataset_path)
    print(f'Loading dataset from {dataset_path}')
    valid_dataset = np.load(dataset_path / 'valid.npy')
    train_dataset = np.load(dataset_path / 'train.npy')
    print(f'Train dataset has {len(train_dataset):,} tokens')
    print(f'Validation dataset has {len(valid_dataset):,} tokens')

    if torch.cuda.is_available():
        device = torch.device('cuda', index=device_id)
    else:
        device = torch.device('cpu')
    model = Model(hparams).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_meter = AverageMeter()
    cudnn.benchmark = True

    if device_id is not None:
        print(f'device {device} initializing process group')
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = master_addr
        torch.distributed.init_process_group(
            backend='nccl', rank=device_id, world_size=n_devices)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device_id], output_device=device_id)
        print(f'process group for {device} initialized')

    def train_step():
        context = _gen_batch(
            train_dataset, n_ctx=n_ctx, batch_size=batch_size * accum_gradients)
        context = torch.LongTensor(context)
        optimizer.zero_grad()
        for ctx in torch.split(context, batch_size):
            ctx = ctx.to(device=device)
            logits = model(ctx)['logits']
            loss = loss_fn(input=logits[:, :-1].reshape([-1, logits.shape[-1]]),
                           target=ctx[:, 1:].reshape(-1))
            loss.backward()
            loss_meter.update(float(loss.item()))
        optimizer.step()

    def save():
        if not is_main:
            return
        model_path = run_path / 'model.pt'
        optim_path = run_path / 'optim.pt'
        for path in [model_path, optim_path]:
            if path.exists():
                shutil.copy(path, f'{path.stem}-prev{path.suffix}')
        torch.save({'state_dict': model.state_dict(), 'step': step}, model_path)
        torch.save(optimizer.state_dict(), optim_path)

    step = 1
    step_tokens = n_ctx * batch_size * accum_gradients * world_size
    epoch_size = len(train_dataset) // step_tokens
    try:
        for epoch in tqdm.trange(1, epochs + 1, desc='epoch',
                                 dynamic_ncols=True):
            epoch_pbar = tqdm.trange(epoch_size, desc=f'epoch {epoch}',
                                     dynamic_ncols=True)
            for _ in epoch_pbar:
                if step % save_every == 0:
                    save()
                if max_steps and step >= max_steps:
                    print(f'max_steps {max_steps} reached, saving and exiting')
                    save()
                    return
                train_step()
                step += 1
                epoch_pbar.set_postfix({
                    'step': step,
                    'loss': f'{loss_meter.mean():.2f}'})
                if step % log_every == 0 and is_main:
                    json_log_plots.write_event(
                        run_path,
                        step=step * step_tokens,
                        loss=loss_meter.mean())
                    loss_meter.reset()

    except KeyboardInterrupt:
        if is_main:
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


def _main_mp(i, kwargs):
    """ Wrapper to use with mp.spawn.
    """
    kwargs['device_id'] = i
    return main(**kwargs)


def fire_main():
    fire.Fire(only_allow_defined_args(main))
