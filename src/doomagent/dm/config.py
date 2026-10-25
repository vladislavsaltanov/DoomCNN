from dataclasses import dataclass

import numpy as np

from doomagent.variants import BASE, load_variants

MODES = ('combat', 'chase', 'gear', 'items', 'patrol')  # teacher decisions, in priority order

# vizdoom/bots.cfg order; a match with n bots adds the first n by name
BOT_NAMES = ('Rambo', 'McClane', 'MacGyver', 'Plissken', 'Machete', 'Anderson', 'Leone', 'Predator',
             'Ripley', 'T800', 'Dredd', 'Conan', 'Bond', 'Jones', 'Blazkowicz')



@dataclass(frozen=True)
class DMCfg:
    # deathmatch settings, separate from the defend_the_center Cfg
    frame_skip: int = 4
    stack: int = 4
    height: int = 240          # network input: 2x the first section's 160x120, small bots are
    width: int = 320           # 2x wider (D1 resolution check, decided 2026-10-22)
    render_w: int = 640        # game render: demo, video, labels
    render_h: int = 480
    hud: bool = False
    map: str = 'map02'         # spec v2: full deathmatch with weapons
    noise_p: float = 0.10
    burst_p: float = 0.02      # recording: chance per step with no bot in sight of a detour burst (record.MatchNoise)
    bots: int = 7
    timelimit: float = 2.0     # minutes per match: 4200 tics, 1050 agent steps
    safe_dist: float = 128.0   # no rocket at a target or wall closer than this
    chase_s: float = 1.0       # go where a bot was last seen for this long
    low_health: float = 40.0   # go for a medikit below this
    low_ammo: float = 5.0      # go for ammo of the best weapon below this
    aim_tol: float = 1.0       # fire when the aim point is within this share of the target's half width
    strafe_period: int = 6     # combat strafe: steps one way before turning back
    dodge: bool = True         # step out of a visible projectile's path
    gear_rank: int = 3         # at a life start, go for a weapon while the best in hand ranks below this
    hitscan_tol: float = 0.0   # degrees: hitscan weapons also fire this far off (their spread); 0: off
    track_ahead: bool = False  # turn to where the target will be after this step, not where it is
    barrel_check: bool = False  # no shot with a barrel in the line of fire closer than BARREL_R
    weapon_first: float = 256.0  # with only weapons below gear_rank: fight bots closer than this, else gear up; 0: off (E1 cycle: +1.40 [+0.25, +2.60] on 40 fresh seeds)
    armor_below: float = 0.0   # go for armor below this much armor; 0: off
    retreat_health: float = 0.0  # leave a fight for a medikit below this health; 0: off
    flick: bool = True         # the whole turn on a step's first tic; the teacher fires when the turn ends on target (E3 cycle: +2.88 [+1.45, +4.45] on 40 fresh seeds)
    target: str = 'dist'       # 'dist': nearest bot; 'angle': nearest the crosshair, kept while seen
    run: bool = False          # the player always runs (+cl_run 1), as fast as the bots; set on every game
    autoaim: bool = False      # Doom's vertical autoaim for the player (sv_noautoaim 0): shots over steps
    cheats: bool = False       # sv_cheats 1: lets PPO training move the player (DeathmatchEnv.teleport); never in evaluation
    depth: bool = False        # read the depth buffer for the coarse depth target (dm.auxtargets); recording only
    min_vis_px: float = 50.0   # a bot with fewer pixels on the 640x480 render is not seen (hidden by a wall or a sliver); 0: off (E3 fix: +1.65 [-0.05, +3.15] on 40 fresh seeds)
    seg_len: int = 32          # M1 training windows, steps
    burn_in: int = 8           # steps that only warm up the GRU in windows that start mid-life


DM = DMCfg()

# disjoint from each other and from every defend_the_center set (1000-21000)
TRAIN_SEEDS = tuple(range(30000, 30300))    # R0: 150 matches, 150 more if needed
VARIANT_SEEDS = tuple(range(31000, 32000))  # R1: teacher on train variants
DAGGER_SEEDS = tuple(range(32000, 33000))   # 40 per round, the same for M0 and M1
VAL_SEEDS = tuple(range(35000, 35020))      # Gate A, checkpoint choice, Gate B
FRESH_SEEDS = tuple(range(36000, 36040))    # re-check of a chosen checkpoint
CHECK_SEEDS = tuple(range(37000, 37006))    # graphics variant check
PILOT_SEEDS = tuple(range(38000, 38010))    # D2 pilot dataset
TEST_SEEDS = tuple(range(39000, 39040))     # D7 only
RL_SEEDS = tuple(range(40000, 41000))
CONFIRM_SEEDS = tuple(range(42000, 42040))  # one look at the teacher after the E3 cycle tuned on FRESH_SEEDS

VAL_OOD_MATCHES = 5   # per val variant
TEST_OOD_MATCHES = 8  # per test variant
VARIANTS_DIR = 'variants_dm'

# label and object names on cig MAP02 (probed 2026-10-23); scripts/dm/dm_discover_labels.py fails on any other name
BOT = 'DoomPlayer'  # every bot, and the player itself (DeathmatchEnv filters it out)
WEAPON_ITEMS = {'Chainsaw': 1, 'Shotgun': 3, 'SuperShotgun': 3, 'Chaingun': 4, 'RocketLauncher': 5,
                'PlasmaRifle': 6, 'BFG9000': 7}  # name -> weapon slot
AMMO_ITEMS = {'Clip': 2, 'ClipBox': 2, 'Shell': 3, 'ShellBox': 3, 'RocketAmmo': 5, 'RocketBox': 5, 'Cell': 6,
              'CellPack': 6, 'Backpack': 0}  # name -> slot whose ammo it gives (0: all)
HEALTH_ITEMS = {'Medikit', 'Stimpack', 'Soulsphere'}
ARMOR_ITEMS = {'GreenArmor', 'BlueArmor', 'ArmorBonus'}
PROJECTILES = {'Rocket': 20.0, 'PlasmaBall': 25.0, 'BFGBall': 25.0}  # name -> map units per tic
BARREL = 'ExplosiveBarrel'
OTHER_NAMES = {'DeadDoomPlayer', 'DeadExplosiveBarrel', 'RocketSmokeTrail', 'TeleportFog', 'ItemFog',
               'BulletPuff', 'Blood', 'PlasmaBallExplosion'}
ITEM_KIND = {**{n: 'weapon' for n in WEAPON_ITEMS}, **{n: 'ammo' for n in AMMO_ITEMS},
             **{n: 'health' for n in HEALTH_ITEMS}, **{n: 'armor' for n in ARMOR_ITEMS}}
AMMO_MAX = {1: 1, 2: 200, 3: 50, 4: 200, 5: 50, 6: 300, 7: 300}  # weapon slot -> ammo maximum (no backpack)

# Doom constants, map units and tics
ROCKET_SPEED = 20.0
PLAYER_RADIUS = 16.0
STEP_HEIGHT = 24.0
PLAYER_HEIGHT = 56.0
TIC_RATE = 35


def _with_val_variants(seeds, root):
    return [(BASE, seeds)] + [(v, seeds[:VAL_OOD_MATCHES]) for v in load_variants('val', root)]


def val_plan(root=VARIANTS_DIR):
    return _with_val_variants(VAL_SEEDS, root)


def fresh_plan(root=VARIANTS_DIR):
    return _with_val_variants(FRESH_SEEDS, root)


def test_plans(root=VARIANTS_DIR):
    return {'id': [(BASE, TEST_SEEDS)],
            'ood': [(v, TEST_SEEDS[:TEST_OOD_MATCHES]) for v in load_variants('test', root)]}


def dagger_plan(round_idx, per_round=40, root=VARIANTS_DIR):
    # rounds start at 1; one plan for M0 and M1, so their rounds differ only by the student. Half the matches on
    # the base map (two thirds of the val and fresh matches are there), the rest spread over the train variants
    train = load_variants('train', root)
    variants = [BASE] * max(1, len(train)) + train
    seeds = DAGGER_SEEDS[(round_idx - 1) * per_round:round_idx * per_round]
    assert len(seeds) == per_round, 'DAGGER_SEEDS exhausted'
    pick = np.random.default_rng(round_idx).integers(0, len(variants), len(seeds))
    groups, by_name = {}, {v.name: v for v in variants}
    for s, i in zip(seeds, pick):
        groups.setdefault(variants[i].name, []).append(s)
    return [(by_name[n], tuple(ss)) for n, ss in sorted(groups.items())]
