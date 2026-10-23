import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from doomagent.dm.agent import DMChooser, pick_device
from doomagent.dm.augment import frame_augment
from doomagent.dm.actions import HEAD_SIZES, HEADS
from doomagent.dm.config import DM, VAL_SEEDS, val_plan
from doomagent.dm.data import MatchSegments, MatchSteps, mirror_batch, split_holdout
from doomagent.dm.evaluate import evaluate_plan, summarize
from doomagent.dm.model import DMNet, aux_loss, bc_loss, load_conv, load_model, save_model, split_logits
from doomagent.evaluate import val_score
from doomagent.variants import BASE

EVERY = 25  # live charts: one point per this many batches


def write_json_atomic(path, obj):
    # rewritten in one move so a reader never sees half a file
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh)
    try:
        os.replace(tmp, path)
    except OSError:  # Windows: the dashboard is reading the file; the next update retries
        pass


def _confusions():
    return {h: np.zeros((k, k), np.int64) for h, k in zip(HEADS, HEAD_SIZES)}


def _add_conf(conf, act, pred):
    # act, pred: (N, 4) teacher and network values
    for k, h in enumerate(HEADS):
        np.add.at(conf[h], (act[:, k], pred[:, k]), 1)


def _acc_per_value(conf):
    out = {}
    for h, m in conf.items():
        rows = m.sum(1)
        out[h] = [float(m[i, i] / rows[i]) if rows[i] else None for i in range(len(m))]
    return out


class Live:
    # the dashboard's training charts: loss and accuracy points, confusion of the recent batches per head
    def __init__(self, out, n_batches, epochs=None, keep=4000):
        self.out, self.n_batches, self.epochs, self.keep = str(out), n_batches, epochs, keep
        self.start = time.time()
        self.points, self.total, self.holdout_confusion = [], 0, None
        self._clear()

    def _clear(self):
        self.sum_ce = self.sum_acc = 0.0
        self.sum_head = {h: 0.0 for h in HEADS}
        self.n = 0
        self.conf = _confusions()

    def add(self, epoch, batch, parts, pred, act):
        self.total += 1
        self.sum_ce, self.sum_acc, self.n = self.sum_ce + parts['ce'], self.sum_acc + parts['acc'], self.n + 1
        for h in HEADS:
            self.sum_head[h] += parts[f'acc_{h}']
        _add_conf(self.conf, act, pred)
        if batch % EVERY and batch != self.n_batches:
            return
        self.points = (self.points + [dict(batch=self.total, epoch=epoch, ce=self.sum_ce / self.n,
                                           acc=self.sum_acc / self.n,
                                           **{f'acc_{h}': self.sum_head[h] / self.n for h in HEADS})])[-self.keep:]
        write_json_atomic(self.out + '.progress.json',
                          dict(epoch=epoch, batch=batch, n_batches=self.n_batches, epochs=self.epochs,
                               start=self.start, t=time.time(),
                               points=self.points))
        write_json_atomic(self.out + '.live.json', dict(
            confusion={h: m.tolist() for h, m in self.conf.items()}, acc_per_value=_acc_per_value(self.conf),
            holdout_confusion=self.holdout_confusion, t=time.time()))
        self._clear()


def _aux_targets(batch, dev):
    return {k: v.to(dev) for k, v in batch.items() if k.startswith('aux_')}


def _to_model(batch, memory, dev):
    # (frames (B, T, S, H, W) uint8, vec (B, T, V), action (B, T, 4), mask (B, T)) on the device
    f, v, a = batch['frames'].to(dev), batch['vec'].to(dev), batch['action'].to(dev)
    if memory:
        return f, v, a, (batch['loss'] & batch['valid']).to(dev)
    return f[:, None], v[:, None], a[:, None], torch.ones(a.shape[0], 1, dtype=torch.bool, device=dev)


@torch.no_grad()
def holdout_metrics(model, ds, memory, dev, batch_size):
    model.eval()
    conf, ce_sum, n, all_ok, aux_sum = _confusions(), 0.0, 0, 0, {}
    for batch in DataLoader(ds, batch_size=batch_size, num_workers=0):
        f, v, a, m = _to_model(batch, memory, dev)
        logits, _, aux = model.forward_aux(f, v)
        if aux is not None:
            _, ap = aux_loss(aux, _aux_targets(batch, dev), m)
            for k, x in ap.items():
                aux_sum.setdefault(k, []).append(x)
        loss, _ = bc_loss(logits, a, m)
        k = int(m.sum())
        ce_sum, n = ce_sum + loss.item() * k, n + k
        sel = m.reshape(-1)
        act = a.reshape(-1, a.shape[-1])[sel].cpu().numpy()
        pred = np.stack([lg.argmax(-1).reshape(-1)[sel].cpu().numpy() for lg in split_logits(logits)], 1)
        _add_conf(conf, act, pred)
        all_ok += int((act == pred).all(1).sum())
    model.train()
    out = dict(holdout_ce=ce_sum / max(n, 1), holdout_acc=all_ok / max(n, 1),
               holdout_acc_per_value=_acc_per_value(conf))
    out.update({f'holdout_acc_{h}': float(np.trace(m) / max(m.sum(), 1)) for h, m in conf.items()})
    out.update({f'holdout_{k}': float(np.mean(x)) for k, x in aux_sum.items() if k != 'aux'})
    return out, conf


def train(kind, dirs, out, epochs, lr=3e-4, batch_steps=256, hflip=True, aug=True, eval_every=2,
          val=None, init=None, conv_init=None, seed=0, workers=2, eval_workers=4, max_minutes=None,
          holdout=0.05, clip=1.0, cfg=DM, live_dir=None, aux_w=0.0):
    # behavior cloning of the teacher's labels: M0 on single steps, M1 on windows inside lives;
    # the checkpoint is chosen by frags on the val matches (val=[]: no games, the last one is kept);
    # live_dir: a folder (dm.livetap.LIVE_DIR) where the val matches are shown on the dashboard;
    # aux_w > 0: auxiliary heads (dm.auxtargets) with this weight on their loss
    assert kind in ('m0', 'm1')
    torch.manual_seed(seed)
    dev, memory = pick_device(), kind == 'm1'
    train_files, hold_files = split_holdout(dirs, holdout)
    make = (lambda fs: MatchSegments(fs, cfg.seg_len, cfg.burn_in, cfg.stack)) if memory else \
        (lambda fs: MatchSteps(fs, cfg.stack))
    bs = max(1, batch_steps // cfg.seg_len) if memory else batch_steps
    dl = DataLoader(make(train_files), batch_size=bs, shuffle=True, num_workers=workers)
    hold = make(hold_files) if hold_files else None
    model = DMNet(cfg, memory, aux=aux_w > 0).to(dev)
    if init:
        model.load_state_dict(load_model(init, cfg).state_dict())
    if conv_init:
        load_conv(model, conv_init)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(dl)))
    val = val_plan() if val is None else val
    config = dict(kind=kind, dirs=list(dirs), epochs=epochs, lr=lr, batch_steps=batch_steps, hflip=hflip,
                  aug=aug, init=init, conv_init=conv_init, seed=seed, clip=clip, aux_w=aux_w,
                  val=[(v.name, len(s)) for v, s in val], train_files=train_files,
                  holdout_files=hold_files)
    live, log, best, t0, stop = Live(out, len(dl), epochs), [], -float('inf'), time.time(), False
    for epoch in range(1, epochs + 1):
        sums, n = dict(ce=0.0, acc=0.0, aux=0.0), 0
        for i, batch in enumerate(dl, 1):
            if hflip and torch.rand(()) < 0.5:
                batch = mirror_batch(batch)
            f, v, a, m = _to_model(batch, memory, dev)
            x = f.float() / 255.0
            if aug:
                B, T = x.shape[:2]
                x = frame_augment(x.reshape(B, -1, *x.shape[3:])).reshape(x.shape)
            logits, _, aux = model.forward_aux(x, v)
            loss, parts = bc_loss(logits, a, m)
            parts['aux'] = 0.0
            if aux is not None:
                la, ap = aux_loss(aux, _aux_targets(batch, dev), m)
                loss, parts['aux'] = loss + aux_w * la, ap['aux']
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
            sel = m.reshape(-1)
            pred = np.stack([lg.argmax(-1).reshape(-1)[sel].cpu().numpy() for lg in split_logits(logits)], 1)
            live.add(epoch, i, parts, pred, a.reshape(-1, a.shape[-1])[sel].cpu().numpy())
            sums = {k: sums[k] + parts[k] for k in sums}
            n += 1
            if max_minutes and time.time() - t0 > 60 * max_minutes:
                stop = True
                break
        if epoch % eval_every and epoch != epochs and not stop:
            continue
        row = dict(epoch=epoch, batches=live.total, minutes=(time.time() - t0) / 60,
                   lr=opt.param_groups[0]['lr'], train_ce=sums['ce'] / max(n, 1),
                   train_acc=sums['acc'] / max(n, 1), train_aux=sums['aux'] / max(n, 1))
        if hold is not None:
            metrics, conf = holdout_metrics(model, hold, memory, dev, bs)
            row.update(metrics)
            live.holdout_confusion = {h: c.tolist() for h, c in conf.items()}
        if val:
            state = {k: t.detach().cpu() for k, t in model.state_dict().items()}
            if live_dir:
                from doomagent.dm.livetap import write_plan
                name = os.path.basename(str(out)).split('.pt')[0]
                write_plan(f'{name} · сеть на val · эпоха {epoch}', '', sum(len(s) for _, s in val), live_dir, 'eval')
            res = evaluate_plan(val, DMChooser(state, memory, cfg), cfg, workers=eval_workers, live=live_dir)
            score, per = val_score(res)
            s = summarize(res)
            row.update(val_score=score, per_variant=per, frags=s['frags'], kd_total=s['kd_total'],
                       stuck_share=s['stuck_share']['mean'])
            if score > best:
                best = score
                save_model(model, out)
        else:
            save_model(model, out)
        log.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'holdout_acc_per_value'}), flush=True)
        with open(str(out) + '.log.json', 'w') as fh:
            json.dump(dict(config=config, log=log), fh, indent=1)
        if stop:
            break
    return best


if __name__ == '__main__':  # loader and evaluation processes re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['m0', 'm1'])
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--val', choices=['full', 'base', 'base10', 'none'], default='full')
    ap.add_argument('--init')
    ap.add_argument('--conv-init')
    ap.add_argument('--max-minutes', type=float)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--eval-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-live', action='store_true', help='no val match frames for the dashboard')
    ap.add_argument('--aux', type=float, default=0.0, help='weight of the auxiliary heads loss (0: none)')
    a = ap.parse_args()
    val = {'full': None, 'base': [(BASE, VAL_SEEDS)], 'base10': [(BASE, VAL_SEEDS[:10])], 'none': []}[a.val]
    print('best val_score:', train(a.kind, a.dirs, a.out, a.epochs, a.lr, eval_every=a.eval_every,
                                   val=val, init=a.init, conv_init=a.conv_init, seed=a.seed,
                                   workers=a.workers, eval_workers=a.eval_workers,
                                   max_minutes=a.max_minutes, live_dir=None if a.no_live else 'runs/dm/live',
                                   aux_w=a.aux))
