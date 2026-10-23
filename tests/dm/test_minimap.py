from doomagent.dm.env import DMPrivileged, MapObject
from doomagent.dm.minimap import WEAPON_COLOR, MapView

BOX = [(0.0, 128.0, ((0, 0, 512, 0, True), (512, 0, 512, 512, True), (512, 512, 0, 512, True),
                     (0, 512, 0, 0, True)))]


def test_to_px_keeps_orientation():
    m = MapView(BOX, size=(320, 320), margin=10)
    assert m.to_px(0, 512) == (10, 10) and m.to_px(512, 0) == (310, 310)
    assert m.to_px(400, 100)[0] > m.to_px(100, 100)[0]   # east is right
    assert m.to_px(100, 400)[1] < m.to_px(100, 100)[1]   # north is up


def test_player_dot_is_drawn_at_its_position():
    m = MapView(BOX)
    img = m.draw(DMPrivileged(px=100.0, py=400.0, sectors=BOX))
    x, y = m.to_px(100, 400)
    assert tuple(img[y, x]) == (255, 255, 255)


def test_draw_works_without_a_teacher_and_keeps_the_base():
    m = MapView(BOX)
    before = m.base().copy()
    m.draw(DMPrivileged(px=256.0, py=256.0, sectors=BOX))
    assert (m.base() == before).all()


def test_minimap_draws_weapons_by_slot():
    m = MapView(BOX)
    img = m.draw(DMPrivileged(px=100.0, py=100.0, sectors=BOX, objects=[MapObject(1, 'Chaingun', 300.0, 300.0, 0.0)]))
    x, y = m.to_px(300, 300)
    assert tuple(img[y, x]) == WEAPON_COLOR[4]
