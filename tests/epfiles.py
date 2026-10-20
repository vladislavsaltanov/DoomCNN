import numpy as np


def make_ep(d, seed, T=6, variant='base', vis=0):
    # a synthetic episode in the recorder format; frame t is filled with the value t
    frames = np.stack([np.full((120, 160), t, np.uint8) for t in range(T)])
    masks = np.zeros((T, 120, 160), bool)
    masks[:, 10:20, 32:40] = bool(vis)
    pre = str(d / f'ep_{variant}_{seed}')
    np.save(pre + '.frames.npy', frames)
    np.save(pre + '.masks.npy', np.packbits(masks, axis=-1))
    ammo = np.arange(T - 1, -1, -1).astype(np.float32)  # the last step has no ammo
    np.savez(pre + '.npz', actions=np.arange(T) % 3, vis=np.full(T, vis, np.int64),
             off=np.zeros(T, np.float32), rewards=np.zeros(T, np.float32), ammo=ammo,
             valid=ammo > 0, seed=seed, variant=variant)
