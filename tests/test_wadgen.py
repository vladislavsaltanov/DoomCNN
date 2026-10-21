import struct

import cv2
import numpy as np

from doomagent.wadgen import (cover_sprites, map_textures, read_wad, sprite_lumps, swap_sprites,
                              texture_pwad_lumps, write_wad)


def name8(s):
    return s.encode('ascii').ljust(8, b'\0')


def test_wad_round_trip(tmp_path):
    lumps = [('S_START', b''), ('POSSA1', b'abc'), ('S_END', b'')]
    write_wad(str(tmp_path / 'x.wad'), lumps)
    assert read_wad(str(tmp_path / 'x.wad')) == lumps


def test_map_textures_binary_and_udmf():
    side = struct.pack('<hh', 0, 0) + name8('STARTAN2') + name8('-') + name8('BRICK1') \
        + struct.pack('<h', 0)
    sector = struct.pack('<hh', 0, 128) + name8('FLOOR4_8') + name8('F_SKY1') \
        + struct.pack('<hhh', 160, 0, 0)
    walls, flats = map_textures([('MAP01', b''), ('SIDEDEFS', side), ('SECTORS', sector)])
    assert walls == ['BRICK1', 'STARTAN2'] and flats == ['FLOOR4_8']
    text = (b'sidedef { texturemiddle = "STONE2"; } '
            b'sector { texturefloor = "FLAT5"; textureceiling = "CEIL1_1"; }')
    walls, flats = map_textures([('TEXTMAP', text)])
    assert walls == ['STONE2'] and flats == ['CEIL1_1', 'FLAT5']


def test_sprite_lumps_reads_between_markers():
    lumps = [('PLAYPAL', b'x'), ('S_START', b''), ('POSSA1', b'a'), ('S_END', b''),
             ('ENDOOM', b'y')]
    assert sprite_lumps(lumps) == {'POSSA1': b'a'}


def test_swap_sprites_renames_only_shared_frames():
    sprites = {'POSSA1': b'p', 'POSSB1': b'p', 'TROOA1': b't', 'TROOC1': b't',
               'TROOA2A8': b't2'}
    assert dict(swap_sprites(sprites, 'TROO', 'POSS')) == {'POSSA1': b't', 'POSSA2A8': b't2'}


def test_texture_pwad_has_namespaces_and_decodable_pngs():
    lumps = texture_pwad_lumps(['BRICK1'], ['FLOOR4_8'], np.random.default_rng(0))
    assert [n for n, _ in lumps] == ['TX_START', 'BRICK1', 'TX_END', 'F_START', 'FLOOR4_8',
                                    'F_END']
    wall = cv2.imdecode(np.frombuffer(lumps[1][1], np.uint8), cv2.IMREAD_COLOR)
    flat = cv2.imdecode(np.frombuffer(lumps[4][1], np.uint8), cv2.IMREAD_COLOR)
    assert wall.shape == (128, 128, 3) and flat.shape == (64, 64, 3)


def test_cover_sprites_fills_missing_target_frames_with_donor_last_frame():
    # every frame letter of the target gets a donor lump, so no original frame shows through
    sprites = {'SARGA1': b's', 'SARGB1': b's', 'SARGC0': b's', 'SARGD0': b's',
               'HEADA1': b'h', 'HEADA2A8': b'h2', 'HEADB0': b'hb'}
    assert dict(cover_sprites(sprites, 'HEAD', 'SARG')) == {
        'SARGA1': b'h', 'SARGA2A8': b'h2', 'SARGB0': b'hb', 'SARGC0': b'hb', 'SARGD0': b'hb'}
