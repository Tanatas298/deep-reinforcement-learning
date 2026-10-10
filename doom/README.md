# Doom Agent: Imitation Learning + Double DQN

![Agent playing against bots](assets/gameplay.gif)

## Problem

The agent plays a deathmatch in the 1993 game DOOM against four built-in bots, using [ViZDoom](https://github.com/Farama-Foundation/ViZDoom). It only sees the game screen (128×128 pixels) and chooses one of 8 actions: move, turn, shoot or do nothing. Its score goes up with kills and hits and down with damage taken.

## Solution

1. **Input:** the screen image is combined with two extra views the game provides: a depth map (how far away things are) and an object map (which pixels belong to objects such as enemies). That gives 5 channels in total.
2. **Network:** a convolutional network (three convolution layers followed by two fully connected layers) that scores each of the 8 actions.
3. **Imitation learning first:** the network was trained for 30 epochs to copy the actions in recorded demonstration games that come with the environment, and the version with the best validation accuracy was kept. This gives the agent a sensible starting behavior.
4. **Reinforcement learning after:** the agent then improved by playing on its own with Double DQN, with Huber loss and gradient clipping for stability, for 150 episodes.

One bug cost a lot of time. The recorded games stored the views in a different order than the live game. The network was receiving scrambled input until the order was fixed.

## Result

The agent scored **184.5** on the course's evaluation server. The top grade required 120. In the replay above, it holds a covered position and shoots bots as they cross its view.

## How to run

The notebook uses the [jku.wad](https://github.com/gerkone/jku.wad) environment. A GPU is recommended, for example Google Colab.

```bash
git clone https://github.com/gerkone/jku.wad
cd jku.wad
pip install -r requirements.txt
```

Copy `doom_agent.ipynb` into the `jku.wad` folder and run it. `model.onnx` is the trained agent that was submitted.

## Credits

The environment and the notebook template come from [jku.wad](https://github.com/gerkone/jku.wad) by Gianluca Galletti, under the MIT license (see [LICENSE](LICENSE)). My parts are the network, the imitation-learning training, the Double DQN training and the input setup.
