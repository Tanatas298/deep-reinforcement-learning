# Walking Robot with PPO

![Trained agent walking](assets/walker.gif)

## Problem

[BipedalWalker](https://gymnasium.farama.org/environments/box2d/bipedal_walker/) is a two-legged robot that has to cross uneven ground. The agent controls the motors of its four joints and is rewarded for moving forward. It loses points for the energy it uses and 100 points if it falls. Reaching the end gives a total of about 300 points.

## Solution

The agent is trained with Proximal Policy Optimization (PPO), an actor-critic method:

- **Actor:** a small network (two layers of 128 units) that outputs a probability distribution over motor commands.
- **Critic:** a second network of the same size that estimates how good the current situation is.
- **Advantage estimation:** each action is judged with Generalized Advantage Estimation (GAE). It compares the reward actually received with the critic's expectation. This made training much more stable than plain Monte Carlo returns:

| Setup | Score |
|---|---|
| Monte Carlo returns, 2,000 episodes | ~70 |
| GAE, 2,000 episodes | 217 |
| GAE, 5,000 episodes | 286 |

## Result

The final agent reaches the end of the course in almost every run. It scored 285.7 on the course's evaluation server. With `evaluate.py` it averages about 290 over 20 runs.

## How to run

```bash
pip install -r requirements.txt

python evaluate.py model.onnx --gif walker.gif    # score the trained agent and save a GIF
python train.py                                   # train a new agent (results in runs/ppo)
python evaluate.py runs/ppo/best.pth              # score your own agent
```

On a laptop CPU, training should take roughly 30–60 minutes; this is an estimate extrapolated from a short test run. If installing Box2D fails on macOS, run `brew install swig` first.

`model.onnx` is the agent submitted in the course. `train.py` implements the same method as standalone code.
