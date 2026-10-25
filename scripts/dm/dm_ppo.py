"""E6: PPO with the teacher's hints from a DAgger student (doomagent.dm.ppo, docs/E6_PPO_проект.md).

    python scripts/dm/dm_ppo.py --init runs/dm/dagger_sp_r2.pt --out runs/dm/ppo_ks
    python scripts/dm/dm_ppo.py --out runs/dm/ppo_scratch --scratch --steps 3000000
    python scripts/dm/dm_ppo.py --init runs/dm/dagger_sp_r2.pt --out runs/dm/ppo_ks --reward stuck=-0.02 --temp 0.8

The env processes re-import this file: torch and the package are imported under the main guard only. A run
restarts from its folder's state.pt.
"""
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', help='start model (.pt of dm.train); omit with --scratch')
    ap.add_argument('--out', required=True)
    ap.add_argument('--scratch', action='store_true', help='a fresh model, no anchor, no warm-up, lr 2.5e-4')
    ap.add_argument('--reward', action='append', default=[], metavar='PART=W',
                    help='weight of a reward part (dm.rlenv.PARTS: frag, death, damage, ..., region, region_stay)')
    ap.add_argument('--no-eval', action='store_true', help='no val matches (smoke runs)')
    ap.add_argument('--demo', action='append', default=[], metavar='DIR',
                    help="the teacher's recorded matches mixed into the updates as BC (repeatable)")
    num = dict(steps=int, n_envs=int, horizon=int, warmup=int, lr=float, critic_lr=float, ks=float, ks_end=float,
               ks_steps=int, prior=float, prior_end=float, temp=float, warp_p=float, demo_end=float, ent=float,
               clip=float, epochs=int, mb_windows=int, eval_every=int, ckpt_every=int, eval_workers=int, base_share=float,
               gamma=float, seed=int)
    for k, t in num.items():
        ap.add_argument('--' + k.replace('_', '-'), type=t)
    a = ap.parse_args()
    assert a.init or a.scratch, '--init or --scratch'

    from dataclasses import replace

    from doomagent.dm.ppo import PPOCfg, run
    pc = PPOCfg()
    if a.scratch:
        pc = replace(pc, ks=0.0, ks_end=0.0, prior=0.0, prior_end=0.0, warmup=0, lr=2.5e-4, lr_end=0.1)
    pc = replace(pc, **{k: getattr(a, k) for k in num if getattr(a, k) is not None}, demo_dirs=tuple(a.demo))
    reward = {k: float(w) for k, w in (x.split('=') for x in a.reward)}
    print(run(a.out, None if a.scratch else a.init, pc, reward, plan=[] if a.no_eval else None))
