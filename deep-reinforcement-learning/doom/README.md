# Doom Agent: Imitation Learning + Double DQN

![Agent playing against bots](assets/gameplay.gif)

The agent plays a deathmatch in DOOM (1993) against four built-in bots, using [ViZDoom](https://github.com/Farama-Foundation/ViZDoom). It sees only the game screen (128×128 pixels) and picks one of 8 actions at each step: move, turn, shoot or do nothing. Its reward goes up with kills and hits and down with damage taken.

## Method

- **Input:** the screen image stacked with two extra views from the game: a depth map (how far away things are) and an object map (which pixels belong to enemies and items). 5 channels in total.
- **Network:** a convolutional Q-network (three convolution layers, two fully connected layers) that scores each of the 8 actions.
- **Imitation learning first:** 30 epochs of training to copy the actions in recorded demonstration games, keeping the version with the best validation accuracy. This gives the agent sensible behavior before it plays on its own.
- **Reinforcement learning after:** 150 episodes of Double DQN, with Huber loss and gradient clipping for stability.

The main bug: the recorded games stored the views in a different channel order than the live game, so the network was trained on scrambled input until the order was fixed.

## Result

The agent scored **184.5** in evaluation matches against the four bots. In the replay above, it holds a covered position and shoots bots as they cross its view.

## How to run

The notebook needs the [jku.wad](https://github.com/gerkone/jku.wad) environment. A GPU is recommended, for example Google Colab.

```bash
git clone https://github.com/gerkone/jku.wad
cd jku.wad
pip install -r requirements.txt
```

Copy `doom_agent.ipynb` into the `jku.wad` folder and run it. `model.onnx` is the trained agent.

## Credits

The environment and the notebook template come from [jku.wad](https://github.com/gerkone/jku.wad) by Gianluca Galletti, under the MIT license (see [LICENSE](LICENSE)). The network, both training stages and the input setup are my own work.
