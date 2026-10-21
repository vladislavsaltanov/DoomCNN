import argparse
import json

import torch
from torch.utils.data import DataLoader

from doomagent.agent import StudentChooser
from doomagent.augment import augment
from doomagent.config import VAL_SEEDS, Cfg
from doomagent.data import EpisodeDataset, hflip_batch
from doomagent.evaluate import evaluate_plan, pixel_only, summarize, val_score
from doomagent.model import StudentCNN, bc_loss, seg_target
from doomagent.variants import BASE, val_plan


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def to_device(batch, dev):
    # MPS has no float64, and the collated offsets arrive as float64
    return {k: (v.float() if v.dtype == torch.float64 else v).to(dev) for k, v in batch.items()}


def train(dirs, out, epochs, lam=0.5, lam_seg=0.5, shoot_w=1.0, hflip=True, aug=False,
          eval_every=2, val=None, init=None, lr=3e-4, seed=0, batch_size=128, workers=0,
          eval_workers=1, cfg: Cfg = Cfg()):
    torch.manual_seed(seed)
    dev = pick_device()
    val = val_plan() if val is None else val
    dl = DataLoader(EpisodeDataset(dirs, cfg.stack), batch_size=batch_size, shuffle=True,
                    num_workers=workers)  # workers exit before each evaluation: less memory at once
    model = StudentCNN(cfg).to(dev)
    if init:
        model.load_state_dict(torch.load(init, map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    config = dict(dirs=list(dirs), epochs=epochs, lam=lam, lam_seg=lam_seg, shoot_w=shoot_w,
                  hflip=hflip, aug=aug, init=init, lr=lr, seed=seed,
                  val=[(v.name, len(s)) for v, s in val])
    best, log = -1e9, []
    for epoch in range(1, epochs + 1):
        model.train()
        sums, n = {}, 0
        for batch in dl:
            batch = to_device(batch, dev)
            if hflip and torch.rand(()) < 0.5:
                batch = hflip_batch(batch)
            obs, mask = batch['obs'].float() / 255.0, batch['mask'].bool()
            if aug:
                obs, mask = augment(obs, mask)
            batch['seg'] = seg_target(mask[:, -1], model.seg_hw)
            loss, parts = bc_loss(model(obs), batch, shoot_w, lam, lam_seg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in dict(parts, loss=loss.item()).items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        if epoch % eval_every == 0 or epoch == epochs:
            chooser = (StudentChooser(model.state_dict(), cfg) if eval_workers > 1
                       else pixel_only(model.act))
            res = evaluate_plan(val, chooser, cfg, workers=eval_workers)
            score, per = val_score(res)
            s = summarize(res)
            means = {k: v / max(n, 1) for k, v in sums.items()}  # epoch means, not the last batch
            log.append(dict(epoch=epoch, **means, val_score=score,
                            per_variant=per, kills_per_shot=s['kills_per_shot'],
                            idle_rate=s['idle_rate']))
            print(log[-1], flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), out)
            with open(str(out) + '.log.json', 'w') as fh:
                json.dump(dict(config=config, log=log), fh, indent=1)
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lam', type=float, default=Cfg().lam)
    ap.add_argument('--lam-seg', type=float, default=Cfg().lam_seg)
    ap.add_argument('--shoot-w', type=float, default=1.0)
    ap.add_argument('--no-hflip', action='store_true')
    ap.add_argument('--aug', action='store_true')
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--val', choices=['full', 'base'], default='full',
                    help='full: base plus val variants; base: base graphics only')
    ap.add_argument('--init')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--eval-workers', type=int, default=1,
                    help='processes for the val episodes; results equal the serial run')
    a = ap.parse_args()
    val = [(BASE, VAL_SEEDS)] if a.val == 'base' else None
    print('best val_score:', train(a.dirs, a.out, a.epochs, a.lam, a.lam_seg, a.shoot_w,
                                   not a.no_hflip, a.aug, a.eval_every, val, a.init, a.lr,
                                   workers=a.workers, eval_workers=a.eval_workers))
