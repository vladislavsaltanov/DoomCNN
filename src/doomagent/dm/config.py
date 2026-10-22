import numpy as np

from doomagent.config import Cfg
from doomagent.variants import BASE, load_variants

FORWARD, FORWARD_LEFT, FORWARD_RIGHT, TURN_LEFT, TURN_RIGHT, FINE_LEFT, FINE_RIGHT, \
    STRAFE_LEFT, STRAFE_RIGHT, SHOOT = range(10)
N_ACTIONS = 10
ACTION_NAMES = ('forward', 'forward_left', 'forward_right', 'turn_left', 'turn_right', 'fine_left',
                'fine_right', 'strafe_left', 'strafe_right', 'shoot')
# a horizontally flipped frame swaps every left action with its right one, strafes included
MIRROR = (FORWARD, FORWARD_RIGHT, FORWARD_LEFT, TURN_RIGHT, TURN_LEFT, FINE_RIGHT, FINE_LEFT,
          STRAFE_RIGHT, STRAFE_LEFT, SHOOT)
NOISE_ACTIONS = tuple(a for a in range(N_ACTIONS) if a != SHOOT)  # recording noise never shoots
TURN_SIDE = {FORWARD_LEFT: -1, TURN_LEFT: -1, FINE_LEFT: -1,
             FORWARD_RIGHT: 1, TURN_RIGHT: 1, FINE_RIGHT: 1}
MODES = ('combat', 'chase', 'items', 'patrol')  # teacher decisions, in priority order

# vizdoom/bots.cfg order; a match with n bots adds the first n by name
BOT_NAMES = ('Rambo', 'McClane', 'MacGyver', 'Plissken', 'Machete', 'Anderson', 'Leone', 'Predator',
             'Ripley', 'T800', 'Dredd', 'Conan', 'Bond', 'Jones', 'Blazkowicz')

# D1 teacher grid (reports/d1_grid.md): closer safe distance, shorter chase, no aim margin
DM = Cfg(scenario='deathmatch', safe_dist=128.0, chase_s=1.0, aim_margin=0.0)

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

VAL_OOD_MATCHES = 5   # per val variant
TEST_OOD_MATCHES = 8  # per test variant
VARIANTS_DIR = 'variants_dm'

# label names on cig MAP01 (probed); scripts/dm_discover_labels.py fails on any other name
BOT = 'DoomPlayer'  # every bot, and the player itself (DeathmatchEnv filters it out)
ITEM_KIND = {'Medikit': 'health', 'RocketAmmo': 'ammo', 'RocketBox': 'ammo',
             'GreenArmor': 'armor', 'BlueArmor': 'armor'}
BARREL = 'ExplosiveBarrel'
OTHER_NAMES = {'DeadDoomPlayer', 'DeadExplosiveBarrel', 'Rocket', 'RocketSmokeTrail',
               'TeleportFog', 'ItemFog'}

# Doom constants, map units and tics
ROCKET_SPEED = 20.0
PLAYER_RADIUS = 16.0
STEP_HEIGHT = 24.0
PLAYER_HEIGHT = 56.0
TIC_RATE = 35
HEALTH_SCALE = 100.0  # observation vector: health / 100, rockets / 50
AMMO_SCALE = 50.0


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
    # rounds start at 1; one plan for M0 and M1, so their rounds differ only by the student
    variants = [BASE] + load_variants('train', root)
    seeds = DAGGER_SEEDS[(round_idx - 1) * per_round:round_idx * per_round]
    assert len(seeds) == per_round, 'DAGGER_SEEDS exhausted'
    pick = np.random.default_rng(round_idx).integers(0, len(variants), len(seeds))
    groups = {}
    for s, i in zip(seeds, pick):
        groups.setdefault(int(i), []).append(s)
    return [(variants[i], tuple(ss)) for i, ss in sorted(groups.items())]
