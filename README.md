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

Training, Saving, Loading:
```python
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```

## Reinforcement Learning Tips and Tricks

### General advice when using Reinforcement Learning

- Do hyperparameter tuning.
- Normalize action and observation spaces, best between -1 and 1.
- Reward is important.
- Good results in RL are mostly due to finding appropriate hyperparameters for the algorithm.
- In RL Zoo, the framework can also tune hyperparameters for you!
- Model-free RL methods (all methods in SB3) are very sample inefficient - use a bigger training budget.
- Some policies are stochastic (A2C or PPO), so call `deterministic=True` in the `.predict()` method for better performance.
- Eval the algorithm every `n` episodes (5 or 20). In the eval stage do not use stochastic noise.
- To train faster you can use SBX (SB3+Jax).
- Which algorithm to choose? See the picture:

<img src="pics/algs.png" width="700">

- For discrete actions use: DQN, QR-DQN, PPO, A2C.
- For continuous actions use: SAC, TD3, CrossQ, TQC, PPO, TRPO, A2C, DroQ.

### Tips and Tricks when creating a custom environment

- Normalize your obs space if possible.
- Normalize your action space and make it symmetric if it is continuous between \[-1, 1\]. This is because almost all RL algorithms rely on Gaussian distributrion with $\mu=0, \sigma=1$.
- Start with a shaped reward (informative) and a simplified version of your problem.
- debug with random actions to check if the env works and follows the gym interface
```python
from stable_baselines3.common.env_checker import check_env

env = CustomEnv(arg1, ...)
# It will check your custom environment and output additional warnings if needed
check_env(env)
```
- To check a random agent on your env, do:
```python
env = YourEnv()
obs, info = env.reset()
n_steps = 10
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()
```
- Preserve Markov property.
- Termination due to timeout needs to be handled separately - `trancated=True`.

### Tips and Tricks when implementing an RL algorithm

- read the original paper several times
- read the existing implementations
- reproducibility is achieved via `seed` parameters
- try to have some "sign of life" on a toy problem
- validate by harder and harder envs with optimizing hyperparameters

For continuous actions:
1. Pendulum (easy to solve)
2. HalfCheetahBullet (medium difficulty with local minima and shaped reward)
3. BipedalWalkerHardcore (if it works on that one, then you can have a cookie)

For discrete actions:
1. CartPole-v1 (easy to be better than random agent, harder to achieve maximal performance)
2. LunarLander 
3. Pong (one of the easiest Atari game)
4. other Atari games (e.g. Breakout)

## Examples of SB3 Usage

[SB3 Examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)

## Vectorised Environments

[Vec Envs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#)

## Policy Networks








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