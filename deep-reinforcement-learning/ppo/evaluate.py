import argparse

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from model import ActorCriticNet


def load_policy(path, state_size, action_size):
    """Return a function mapping a state to the agent's best action, from an .onnx or .pth file."""
    if path.endswith(".onnx"):
        import onnxruntime as ort

        session = ort.InferenceSession(path)
        input_name = session.get_inputs()[0].name
        return lambda state: np.tanh(session.run(None, {input_name: state[None].astype(np.float32)})[0][0])

    checkpoint = torch.load(path)
    hidden_size = checkpoint["state_dict"]["actor.fc1.weight"].shape[0]
    model = ActorCriticNet(state_size, action_size, hidden_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def policy(state):
        # use the mean action (no sampling) when evaluating
        with torch.no_grad():
            mu_logits, _ = model.actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            return torch.tanh(mu_logits)[0].numpy()

    return policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent on BipedalWalker-v3.")
    parser.add_argument("model", help="model.onnx or a .pth checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", help="save the first episode as a GIF to this path")
    args = parser.parse_args()

    env = gym.make("BipedalWalker-v3", render_mode="rgb_array" if args.gif else None)
    policy = load_policy(args.model, env.observation_space.shape[0], env.action_space.shape[0])

    scores, frames = [], []
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        done, score, step = False, 0.0, 0
        while not done:
            if args.gif and episode == 0 and step % 3 == 0:
                frames.append(Image.fromarray(env.render()).resize((360, 240)))
            state, reward, terminated, truncated, _ = env.step(policy(state))
            score += reward
            step += 1
            done = terminated or truncated
        scores.append(score)
        print(f"episode {episode + 1:2d}: {score:7.1f}")

    print(f"mean score over {args.episodes} episodes: {np.mean(scores):.1f} (std {np.std(scores):.1f})")
    if args.gif:
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=60, loop=0)


if __name__ == "__main__":
    main()
