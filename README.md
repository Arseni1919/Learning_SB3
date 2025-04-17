# Learning Stable-Baselines3 - Reliable Reinforcement Learning Implementations


## Getting Started

Example run:
```python
import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="rgb_array")
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
```

Same with one line:
```python
model = A2C("MlpPolicy", "CartPole-v1").learn(10000)
```

## Reinforcement Learning Tips and Tricks

- Normalize action and observation spaces, best between -1 and 1
- Reward is important

Which algorithm to choose?

<img src="pics/algs.png" width="700">












## Credits

To cite them:
```bib
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```