# Dataset report: data/train

- episodes: 300, steps: 85064, mean episode length: 283.5
- action share (turn_left / turn_right / shoot / fine_left / fine_right): 0.270 / 0.258 / 0.173 / 0.158 / 0.141
- action share on steps with ammo, which the action loss sees: 0.300 / 0.285 / 0.098 / 0.169 / 0.148
- share of steps with ammo (valid): 0.862
- share of steps with a visible enemy: 0.731
- episodes per variant: {"base": 300}
