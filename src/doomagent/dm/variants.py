"""Graphics variants of a cig.wad map (MAP02 since spec v2): new wall and floor textures (tex), the bots drawn as another
monster (spr), or both (mix). Bots are DoomPlayer actors, drawn with the PLAY sprite, so only PLAY is
replaced; the player's weapon, the rockets and the items keep their look. Built on the first
section's wadgen helpers, which stay unchanged.
"""
import json
import os

import numpy as np

from doomagent.wadgen import cover_sprites, map_textures, read_wad, sprite_lumps, texture_pwad_lumps, write_wad

TARGET = 'PLAY'  # bots and their corpses
SPR_DONORS = {'train': ('TROO', 'SPOS', 'POSS', 'BOS2'), 'val': ('CPOS',), 'test': ('VILE', 'SSWV')}
MIX_DONORS = {'train': ('SARG', 'TROO', 'SPOS', 'POSS'), 'val': ('CPOS',), 'test': ('VILE', 'SSWV')}
N_TEX = {'train': 8, 'val': 2, 'test': 2}


def map_lumps(lumps, name='MAP01'):
    # the lumps of one map: from its marker to ENDMAP (UDMF) or to the next map marker (binary)
    out, inside = [], False
    for n, b in lumps:
        if inside and (n == 'ENDMAP' or _is_marker(n)):
            if n == 'ENDMAP':
                out.append((n, b))
            break
        if n == name:
            inside = True
        if inside:
            out.append((n, b))
    return out


def _is_marker(n):
    return (len(n) == 5 and n.startswith('MAP') and n[3:].isdigit()) or \
        (len(n) == 4 and n[0] == 'E' and n[2] == 'M' and n[1].isdigit() and n[3].isdigit())


def sprite_block(sprites, donor):
    return [('S_START', b'')] + cover_sprites(sprites, donor, TARGET) + [('S_END', b'')]


def source_lumps(map_name=None):
    # the textures and flats of one cig.wad map (the deathmatch map by default) and freedoom2's sprites
    import vizdoom as vzd

    from doomagent.dm.config import DM
    name = (map_name or DM.map).upper()
    walls, flats = map_textures(map_lumps(read_wad(os.path.join(vzd.scenarios_path, 'cig.wad')), name))
    sprites = sprite_lumps(read_wad(os.path.join(os.path.dirname(vzd.__file__), 'freedoom2.wad')))
    return walls, flats, sprites


def make_variants(out, seed=0, splits=('train', 'val', 'test'), kinds=('tex', 'spr', 'mix'), map_name=None):
    # writes <out>/<name>.wad and <name>.json per variant; returns the names
    walls, flats, sprites = source_lumps(map_name)
    for d in {d for ds in (SPR_DONORS, MIX_DONORS) for v in ds.values() for d in v}:
        assert any(n.startswith(d) for n in sprites), f'no {d} sprites in freedoom2.wad'
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    names = []

    def save(name, split, kind, lumps):
        path = f'{out}/{name}.wad'  # forward slash: the json is shared across OSes
        write_wad(path, lumps)
        with open(os.path.join(out, name + '.json'), 'w') as fh:
            json.dump(dict(name=name, split=split, kind=kind, iwad='', pwads=[path]), fh, indent=1)
        names.append(name)

    for split in splits:
        if 'tex' in kinds:
            for i in range(N_TEX[split]):
                save(f'tex_{split}_{i:02d}', split, 'tex', texture_pwad_lumps(walls, flats, rng))
        if 'spr' in kinds:
            for d in SPR_DONORS[split]:
                save(f'spr_{split}_{d.lower()}', split, 'spr', sprite_block(sprites, d))
        if 'mix' in kinds:
            for i, d in enumerate(MIX_DONORS[split]):
                save(f'mix_{split}_{i:02d}_{d.lower()}', split, 'mix',
                     texture_pwad_lumps(walls, flats, rng) + sprite_block(sprites, d))
    return names
