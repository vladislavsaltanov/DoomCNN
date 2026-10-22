"""ViZDoom seeds that really give different games.

DoomGame.set_seed(s) seeds a std::mt19937 and hands its first output to the engine, which reads it
as a signed int: on Windows every output above INT_MAX becomes INT_MAX, so about half of all seeds
play one and the same game (measured 2026-10-22: 14 of the 30 defend_the_center val seeds were one
episode; in deathmatch 16 of 24 seeds were one match). vizdoom_seed(s) picks, for every seed s, a
seed whose first output fits, from a block of 64 that belongs to s alone.
"""
INT_MAX = 2 ** 31 - 1
BLOCK = 64


def first_mt19937(seed):
    # the first number of std::mt19937(seed): init_genrand, one twist, tempering
    mt = [seed & 0xffffffff]
    for i in range(1, 624):
        mt.append((1812433253 * (mt[-1] ^ (mt[-1] >> 30)) + i) & 0xffffffff)
    y = (mt[0] & 0x80000000) | (mt[1] & 0x7fffffff)
    y = mt[397] ^ (y >> 1) ^ (0x9908b0df if y & 1 else 0)
    y ^= y >> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >> 18
    return y & 0xffffffff


def vizdoom_seed(seed):
    for k in range(BLOCK):
        s = seed * BLOCK + k
        if first_mt19937(s) <= INT_MAX:
            return s
    raise ValueError(f'no usable ViZDoom seed for {seed}')
