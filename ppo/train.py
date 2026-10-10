import csv
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ActorCriticNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
seed = 31
episodes = 5000           # run agent for this many episodes
max_episode_steps = 1600  # episode time limit (the full BipedalWalker course)
epochs = 10               # run n epochs of network updates
hidden_size = 128         # number of units in NN hidden layers
learning_rate = 3e-4      # learning rate for optimizer
discount = 0.99           # discount factor gamma value
gae_lambda = 0.95         # GAE lambda

batch_size = 128          # number of samples used for an update
min_transitions = 2048    # number of minimum transitions until update is triggered
capacity = 10000          # maximum number of transitions stored in the rollout buffer

eps_clip = 0.2                   # clipping for importance sampling updates
loss_scales = (0.5, 1.0, 0.005)  # loss scales (value loss, policy loss, entropy loss)
betas = (0.9, 0.999)             # optimizer beta parameters
weight_decay = 0.0               # optimizer weight decay
checkpoint_interval = 100        # save last.pth every n episodes

out_dir = "runs/ppo"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def create_env(env_id="BipedalWalker-v3", hardcore=False):
    env = gym.make(env_id, max_episode_steps=max_episode_steps, hardcore=hardcore)
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    return env, action_size, state_size


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transforms(state):
    # transform numpy to tensor and push to device
    return torch.FloatTensor(state).to(device)


# ---------------------------------------------------------------------------
# Training buffer
# ---------------------------------------------------------------------------
class Transition(object):
    """Transition helper object
    """
    def __init__(self, state, action, reward, next_state, log_probs):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.log_probs = log_probs
        self.advantage = 0.0  # filled in by RolloutBuffer.add
        self.target = 0.0     # value target for the critic, filled in by RolloutBuffer.add


class Episode(object):
    """Class for collecting an episode of transitions
    """
    def __init__(self):
        self.transitions = []
        self.total_reward = 0.0

    def size(self):
        return len(self.transitions)

    def append(self, transition):
        self.transitions.append(transition)


class BufferDataset(Dataset):
    """Buffer dataset used to iterate over buffer samples when training.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = self.data[idx]
        return t.state, t.action, t.log_probs, t.advantage, t.target


class RolloutBuffer(object):
    """Buffer to collect samples while rolling out in the environment.
    """
    def __init__(self, capacity, batch_size, min_transitions):
        self.capacity = capacity
        self.batch_size = batch_size
        self.min_transitions = min_transitions
        self.buffer = []
        self.position = 0

    def add(self, episode, value_fn, terminated, discount, gae_lambda):
        # compute advantages with GAE, then store the transitions
        transitions = episode.transitions
        T = len(transitions)

        rewards_arr = np.array([t.reward for t in transitions], dtype=np.float32)
        states_arr = np.array([t.state for t in transitions], dtype=np.float32)
        next_states_arr = np.array([t.next_state for t in transitions], dtype=np.float32)

        with torch.no_grad():
            values = value_fn(transforms(states_arr)).squeeze(-1).cpu().numpy()
            next_values = value_fn(transforms(next_states_arr)).squeeze(-1).cpu().numpy()

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for i in range(T - 1, -1, -1):
            if i == T - 1:
                # no bootstrapping after a real terminal state, only after the time limit
                v_bootstrap = 0.0 if terminated else next_values[i]
            else:
                v_bootstrap = next_values[i]
            delta = rewards_arr[i] + discount * v_bootstrap - values[i]
            gae = delta + discount * gae_lambda * gae
            advantages[i] = gae

        targets = advantages + values  # TD(lambda) return target for the critic

        for i, t in enumerate(transitions):
            t.advantage = float(advantages[i])
            t.target = float(targets[i])

        for t in transitions:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = t
            self.position = (self.position + 1) % self.capacity

    def reset(self):
        del self.buffer[:]
        self.position = 0

    def create_dataloader(self):
        return DataLoader(BufferDataset(self.buffer), batch_size=self.batch_size, shuffle=True)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent(object):
    """Agent class used for training, saving data and handling the model.
    """
    def __init__(self, buffer, state_size, action_size, hidden_size, learning_rate, eps_clip, n_epochs,
                 weight_decay, betas, loss_scales, checkpoint_dir):
        self.action_size = action_size
        self.state_size = state_size
        self.buffer = buffer
        self.checkpoint_dir = checkpoint_dir
        self.loss_scales = loss_scales
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        self.model = ActorCriticNet(state_size, action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas,
                                    weight_decay=weight_decay)

    def save_checkpoint(self, epoch, info="last"):
        """Saves a model checkpoint"""
        state = {
            "info": info,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(state, os.path.join(self.checkpoint_dir, f"{info}.pth"))

    def save_onnx_checkpoint(self, file_name):
        """Exports the actor to ONNX"""
        dummy_input = torch.randn((1, self.state_size)).to(device)
        self.model.actor.eval()
        torch.onnx.export(self.model.actor, dummy_input, file_name, input_names=["state"],
                          output_names=["mu_logits", "sigma_logits"], dynamo=False)

    def act(self, state):
        state = transforms(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            action, log_probs = self.model.act(state)
        return action, log_probs

    def train(self):
        for i in range(self.n_epochs):
            # create a dataloader based on the current buffer
            loader = self.buffer.create_dataloader()
            # iterate over the samples in the dataloader
            for states, actions, old_log_probs, advantage, target in loader:
                states = states.float().to(device)
                actions = actions.float().to(device)
                old_log_probs = old_log_probs.float().to(device)
                advantage = advantage.float().to(device)
                target = target.float().to(device)

                if advantage.numel() > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                new_log_probs, entropy, state_value = self.model.evaluate(states, actions)
                state_value = state_value.squeeze(-1)

                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                policy_loss = -torch.min(surr1, surr2)

                value_loss = F.mse_loss(state_value, target, reduction="none")

                value_scale, policy_scale, entropy_scale = self.loss_scales
                loss = policy_scale * policy_loss + value_scale * value_loss - entropy_scale * entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------
class Runner(object):
    """Runner class performing the rollout in the environment and calling the agent training function periodically
    """
    def __init__(self, env, agent, buffer, n_episodes, discount, gae_lambda, checkpoint_interval, log_file):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.n_episodes = n_episodes
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.checkpoint_interval = checkpoint_interval
        self.log_file = log_file
        self.stats_rewards_list = []
        self.max_reward = -np.inf

    def run(self):
        self.agent.model.eval()
        log = csv.writer(self.log_file)
        log.writerow(["episode", "reward", "length"])

        with tqdm(range(self.n_episodes)) as pbar:
            for e in pbar:
                state, info = self.env.reset()
                episode = Episode()

                if e % self.checkpoint_interval == 0:
                    self.agent.save_checkpoint(e)

                done = False
                while not done:
                    action, log_probs = self.agent.act(state)
                    action_np = action.squeeze(0).cpu().numpy()

                    next_state, reward, terminated, truncated, info = self.env.step(action_np)
                    episode.total_reward += reward

                    transition = Transition(
                        state=np.asarray(state, dtype=np.float32),
                        action=action_np.astype(np.float32),
                        reward=float(reward),
                        next_state=np.asarray(next_state, dtype=np.float32),
                        log_probs=log_probs.squeeze(0).cpu(),
                    )
                    episode.append(transition)
                    state = next_state
                    done = terminated or truncated

                self.buffer.add(episode, value_fn=self.agent.model.critic, terminated=terminated,
                                discount=self.discount, gae_lambda=self.gae_lambda)

                # update the agent when the buffer has enough samples, then empty it
                if len(self.buffer) >= self.buffer.min_transitions:
                    self.agent.model.train()
                    self.agent.train()
                    self.agent.model.eval()
                    self.buffer.reset()

                self.stats_rewards_list.append((e, episode.total_reward, episode.size()))
                log.writerow([e + 1, round(episode.total_reward, 2), episode.size()])
                self.log_file.flush()

                # save best model
                if episode.total_reward > self.max_reward:
                    self.max_reward = episode.total_reward
                    self.agent.save_checkpoint(e, "best")

                recent = [r[1] for r in self.stats_rewards_list[-10:]]
                pbar.set_postfix({"r": f"{episode.total_reward:.1f}",
                                  "avg10": f"{np.mean(recent):.1f}",
                                  "max": f"{self.max_reward:.1f}"})

        self.agent.save_checkpoint(self.n_episodes)


if __name__ == "__main__":
    env, action_size, state_size = create_env()
    set_seed(seed)
    env.reset(seed=seed)

    buffer = RolloutBuffer(capacity=capacity, batch_size=batch_size, min_transitions=min_transitions)
    agent = Agent(buffer=buffer, state_size=state_size, action_size=action_size, hidden_size=hidden_size,
                  learning_rate=learning_rate, eps_clip=eps_clip, n_epochs=epochs, weight_decay=weight_decay,
                  betas=betas, loss_scales=loss_scales, checkpoint_dir=out_dir)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as log_file:
        runner = Runner(env=env, agent=agent, buffer=buffer, n_episodes=episodes, discount=discount,
                        gae_lambda=gae_lambda, checkpoint_interval=checkpoint_interval, log_file=log_file)
        runner.run()

    agent.save_onnx_checkpoint(os.path.join(out_dir, "model.onnx"))
    print(f"Done. Best episode reward: {runner.max_reward:.1f}. Saved to {out_dir}")
