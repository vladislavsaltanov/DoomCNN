import json

import numpy as np
import pytest

from doomagent.dm.variants import TARGET, make_variants, map_lumps, source_lumps, sprite_block
from doomagent.wadgen import read_wad


def test_map_lumps_takes_one_map():
    lumps = [('DECORATE', b'd'), ('MAP01', b''), ('TEXTMAP', b'one'), ('ENDMAP', b''),
             ('MAP02', b''), ('TEXTMAP', b'two'), ('ENDMAP', b'')]
    assert map_lumps(lumps) == [('MAP01', b''), ('TEXTMAP', b'one'), ('ENDMAP', b'')]
    binary = [('MAP01', b''), ('THINGS', b't'), ('SIDEDEFS', b's'), ('MAP02', b''), ('THINGS', b'u')]
    assert map_lumps(binary) == binary[:3]


def test_cig_map01_textures_only():
    walls, flats, _ = source_lumps()
    assert walls and flats and 'F_SKY1' not in flats


def test_sprite_block_targets_play_only():
    _, _, sprites = source_lumps()
    block = sprite_block(sprites, 'TROO')
    assert block[0][0] == 'S_START' and block[-1][0] == 'S_END'
    inner = [n for n, _ in block[1:-1]]
    assert inner and all(n.startswith(TARGET) for n in inner)
    # every PLAY frame letter is covered, so no frame of the original player shows through
    assert {n[4] for n in inner} >= {n[4] for n in sprites if n.startswith(TARGET)}


def test_make_variants_writes_json_and_wad(tmp_path):
    names = make_variants(str(tmp_path), splits=('val',))
    assert len(names) == 4  # 2 tex + 1 spr + 1 mix
    d = json.load(open(tmp_path / f'{names[0]}.json'))
    assert d['split'] == 'val' and d['pwads'][0].endswith(f'/{names[0]}.wad')
    assert read_wad(d['pwads'][0])[0][0] == 'TX_START'


@pytest.mark.vizdoom
def test_a_tex_variant_changes_the_frames(tmp_path):
    from doomagent.dm.config import DM, FORWARD
    from doomagent.dm.env import DeathmatchEnv
    from doomagent.variants import BASE, load_variants
    make_variants(str(tmp_path), splits=('val',), kinds=('tex',))
    v = load_variants(root=str(tmp_path))[0]
    frames = []
    for var in (BASE, v):
        env = DeathmatchEnv(DM, var)
        obs = env.reset(35000)
        for _ in range(5):
            obs, _, _ = env.step(FORWARD)
        frames.append(obs.frames[-1].astype(np.float32))
        env.close()
    assert np.abs(frames[0] - frames[1]).mean() > 8
