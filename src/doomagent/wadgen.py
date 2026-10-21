import re
import struct

import cv2
import numpy as np


def read_wad(path):
    with open(path, 'rb') as fh:
        raw = fh.read()
    n, ofs = struct.unpack_from('<ii', raw, 4)
    out = []
    for i in range(n):
        pos, size, name = struct.unpack_from('<ii8s', raw, ofs + 16 * i)
        out.append((_name(name), raw[pos:pos + size]))
    return out


def write_wad(path, lumps):
    data, entries, pos = bytearray(), [], 12
    for name, b in lumps:
        entries.append((pos, len(b), name))
        data += b
        pos += len(b)
    with open(path, 'wb') as fh:
        fh.write(b'PWAD' + struct.pack('<ii', len(lumps), pos))
        fh.write(bytes(data))
        for p, size, name in entries:
            fh.write(struct.pack('<ii', p, size) + name.encode('ascii').ljust(8, b'\0')[:8])


def _name(b):
    return b.split(b'\0')[0].decode('ascii', 'replace').upper()


def map_textures(lumps):
    # wall textures and flats used by the map, from binary SIDEDEFS/SECTORS or a UDMF TEXTMAP
    walls, flats = set(), set()
    for name, b in lumps:
        if name == 'SIDEDEFS':
            for k in range(0, len(b) - 29, 30):
                for o in (4, 12, 20):
                    walls.add(_name(b[k + o:k + o + 8]))
        elif name == 'SECTORS':
            for k in range(0, len(b) - 25, 26):
                flats.add(_name(b[k + 4:k + 12]))
                flats.add(_name(b[k + 12:k + 20]))
        elif name == 'TEXTMAP':
            t = b.decode('ascii', 'replace')
            walls |= {w.upper() for w in
                      re.findall(r'texture(?:top|middle|bottom)\s*=\s*"([^"]+)"', t, re.I)}
            flats |= {f.upper() for f in
                      re.findall(r'texture(?:floor|ceiling)\s*=\s*"([^"]+)"', t, re.I)}
    return sorted(walls - {'-', ''}), sorted(flats - {'F_SKY1', ''})


def sprite_lumps(lumps):
    out, inside = {}, False
    for name, b in lumps:
        if name in ('S_START', 'SS_START'):
            inside = True
        elif name in ('S_END', 'SS_END'):
            inside = False
        elif inside and len(name) >= 6:
            out[name] = b
    return out


def swap_sprites(sprites, donor, target):
    # donor frames renamed to the target sprite; only frame letters the target already has,
    # so the engine never sees a sprite with frames it does not expect
    frames = {n[4] for n in sprites if n.startswith(target)}
    return [(target + n[4:], b) for n, b in sorted(sprites.items())
            if n.startswith(donor) and n[4] in frames]


def cover_sprites(sprites, donor, target):
    # swap_sprites plus the target's frame letters the donor lacks, drawn with the donor's last
    # frame (usually its corpse), so no frame of the original monster shows through
    out = swap_sprites(sprites, donor, target)
    have = {n[4] for n, _ in out}
    last = max(n[4] for n in sprites if n.startswith(donor))
    tail = [(n[4:], b) for n, b in sorted(sprites.items()) if n.startswith(donor) and n[4] == last]
    for letter in sorted({n[4] for n in sprites if n.startswith(target)} - have):
        out += [(target + letter + rest[1:].replace(last, letter), b) for rest, b in tail]
    return out


def random_texture_rgb(rng, w, h):
    c0, c1 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    yy, xx = np.mgrid[0:h, 0:w]
    kind = int(rng.integers(0, 4))
    if kind == 0:  # smooth noise
        cells = int(rng.integers(2, 12))
        t = cv2.resize(rng.random((cells, cells)), (w, h), interpolation=cv2.INTER_CUBIC)
    elif kind == 1:  # stripes
        th, f = rng.uniform(0, np.pi), rng.uniform(2, 12)
        t = 0.5 + 0.5 * np.sin(2 * np.pi * f * (xx * np.cos(th) + yy * np.sin(th)) / max(w, h))
    elif kind == 2:  # checker
        s = int(rng.integers(4, 33))
        t = ((xx // s + yy // s) % 2).astype(float)
    else:  # bricks
        bh, bw = int(rng.integers(8, 33)), int(rng.integers(16, 65))
        t = ((((xx + (yy // bh % 2) * bw // 2) % bw) < 3) | (yy % bh < 3)).astype(float)
    t = np.clip(t, 0.0, 1.0)[..., None]
    rgb = c0 * (1 - t) + c1 * t + rng.normal(0, 8, (h, w, 3))
    return rgb.clip(0, 255).astype(np.uint8)


def _png(rgb):
    return cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()


def texture_pwad_lumps(walls, flats, rng):
    # ZDoom takes PNG graphics in the TX_ (textures) and F_ (flats) namespaces of a PWAD
    lumps = [('TX_START', b'')] + [(n, _png(random_texture_rgb(rng, 128, 128))) for n in walls]
    lumps += [('TX_END', b''), ('F_START', b'')]
    lumps += [(n, _png(random_texture_rgb(rng, 64, 64))) for n in flats] + [('F_END', b'')]
    return lumps
