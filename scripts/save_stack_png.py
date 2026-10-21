import os

from doomagent.config import TURN_RIGHT
from doomagent.env import DoomEnv
from doomagent.viz import stack_image

os.makedirs('artifacts', exist_ok=True)
env = DoomEnv()
obs = env.reset(5000)
for _ in range(5):
    obs, _, _ = env.step(TURN_RIGHT)
stack_image(obs, 'artifacts/s0_stack.png')
env.close()
