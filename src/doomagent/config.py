from dataclasses import dataclass

TURN_LEFT, TURN_RIGHT, SHOOT, FINE_LEFT, FINE_RIGHT = 0, 1, 2, 3, 4
N_ACTIONS = 5
ACTION_NAMES = ('turn_left', 'turn_right', 'shoot', 'fine_left', 'fine_right')

TRAIN_SEEDS = tuple(range(1000, 1300))        # R0: teacher on the base graphics
TRAIN_SEEDS_EXTRA = tuple(range(1300, 1600))  # R0: extra recording if needed
DAGGER_SEEDS = tuple(range(2000, 3000))       # R2+: 100 per round
VARIANT_SEEDS = tuple(range(3000, 4000))      # R1: teacher on train variants
VAL_SEEDS = tuple(range(5000, 5030))
TEST_SEEDS = tuple(range(9000, 9100))
RL_SEEDS = tuple(range(20000, 21000))

VAL_OOD_EPISODES = 10   # per val variant
TEST_OOD_EPISODES = 20  # per test variant


@dataclass(frozen=True)
class Cfg:
    frame_skip: int = 4
    stack: int = 4
    height: int = 120          # network input
    width: int = 160
    render_w: int = 640        # game render: demo, video, labels
    render_h: int = 480
    hud: bool = False
    turn_mode: str = 'binary'  # 'binary' | 'delta', coarse turns, fixed in S1
    turn_delta: float = 4.0    # TURN_LEFT_RIGHT_DELTA value in delta mode, fixed in S1
    fine_turn_delta: float = 0.5  # FINE_LEFT/FINE_RIGHT value per tic: 2 deg per step, best of the S1 grid
    fine_zone: float = 0.15    # teacher steps finely when |offset| is within this (about 8.5 deg)
    aim_margin: float = 0.2    # teacher shoots when the crosshair is inside the box shrunk by this share per side
    min_target_w: float = 0.0  # px at network resolution; narrower targets are tracked but not shot
    noise_p: float = 0.10
    lam: float = 0.5           # visibility + offset heads
    lam_seg: float = 0.5       # enemy mask head
    scenario: str = 'defend_the_center'  # 'defend_the_center' | 'deathmatch'
    # deathmatch only (cig.wad MAP01 against ViZDoom bots); defend_the_center ignores these
    bots: int = 7
    timelimit: float = 2.0     # minutes per match: 4200 tics, 1050 agent steps at frame_skip 4
    safe_dist: float = 160.0   # teacher: no rocket at a target, wall or barrel closer than this
    chase_s: float = 2.0       # teacher: go where a bot was last seen for this long
    low_health: float = 50.0   # teacher: go for a known medikit below this
    low_ammo: float = 5.0      # teacher: go for known rockets below this
    seg_len: int = 32          # M1 training windows, steps
    burn_in: int = 8           # steps that only warm up the GRU in windows that start mid-life
