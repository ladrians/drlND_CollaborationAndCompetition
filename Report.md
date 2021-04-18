# Collaboration and Competition

My solution for the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Collaboration and Competition project, following the [Rubric](https://review.udacity.com/#!/rubrics/1891/view) for the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/release_4_branch/docs/Learning-Environment-Examples.md#tennis) environment.

[//]: # (Image References)

[image1]: ./extra/train01.png
[image2]: ./extra/cc01.gif

---
## Description

Two agents control rackets to bounce a ball over a net to play Tennis.

If an agent hits the ball over the net, it receives a reward of `+0.1`.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of `-0.01`; the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of `+0.5` (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least `+0.5`.

## Architecture

The initial configuration reuses the [ContinuousControl](https://github.com/ladrians/drlND_ContinuousControl) in particular the [DDPG algorithm](https://arxiv.org/abs/1509.02971) adapted for Multi-agent usage, based on the MADDPG Lab task.

Both agent states and actions are combined for training. I used a vanilla deep neural network consisting of 3 fully connected layers for the `actor` and `critic`. The `actor` approximates the best action in the given state while the `critic` approximates the associated `Q-value `. Used 200x150 hidden units. In both model changed the usage of `relu` to `leaky_relu` since it performed better (from solving the scenario in to `2093` to `1375` episodes).

### Hyperparameters

The initial configuration was set equal to the previous project, which generated unconclusive results, it was detected unstability during training. In several iterations the following parameters where modified. The final values are:

```python
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3 
WEIGHT_DECAY = 1e-6
SIGMA = 0.15
NOISE_DECAY = 0.9999

fc1_units = 200
fc2_units = 150
```

The following parameter changes were tested based on a try and error basis:

 * `BATCH_SIZE`: `32`, `64`, `128`, `256` using a default of `256`; small batch sizes did not get good results, kept enlarging it got got better results.
 * `LR_CRITIC`: `1e-4` to `1e-3`; learning was extremely erratic, with minor changes on the learning rate of the critic got better results.
 * `NOISE_DECAY`: from `0.` to `0.9999`; not sure if this parameter changes was representative or not.
 * `fc1_units`x`fc2_units`: from `400x300`, `256x128` to `200x150`.

## Training

All training was done on the [Tennis.ipynb](Tennis.ipynb) notebook, taking as reference the [DDPG algorithm](https://arxiv.org/abs/1509.02971) modified from the [ContinuousControl](https://github.com/ladrians/drlND_ContinuousControl) project for multi-agent usage, where the training is done on both agents at the same time.

A wrapper `MDDPGAgent` class was created to encapsulate the agents and to act on both on every iteration.

The result (local execution) is as follows:

```sh
# relu
Training started using 4000 episodes and 800 steps
Episode 100	Average Score: 0.01	Score: 0.050
Episode 200	Average Score: -0.00	Score: -0.00
Episode 300	Average Score: -0.00	Score: -0.00
Episode 400	Average Score: -0.00	Score: -0.00
Episode 500	Average Score: -0.00	Score: -0.00
Episode 600	Average Score: 0.02	Score: 0.0500
Episode 700	Average Score: 0.02	Score: 0.050
Episode 800	Average Score: 0.02	Score: -0.00
Episode 900	Average Score: 0.02	Score: 0.050
Episode 1000	Average Score: 0.02	Score: -0.00
Episode 1100	Average Score: 0.02	Score: 0.050
Episode 1200	Average Score: 0.01	Score: -0.00
Episode 1300	Average Score: 0.03	Score: 0.050
Episode 1400	Average Score: 0.03	Score: 0.150
Episode 1500	Average Score: 0.03	Score: 0.100
Episode 1600	Average Score: 0.03	Score: 0.050
Episode 1700	Average Score: 0.05	Score: 0.150
Episode 1800	Average Score: 0.06	Score: 0.150
Episode 1900	Average Score: 0.13	Score: 0.200
Episode 2000	Average Score: 0.30	Score: 0.050
Episode 2093	Average Score: 0.51	Score: 1.550
Environment solved in 2093 episodes!	Average Score: 0.51
# leaky_relu
Training started using 4000 episodes and 800 steps
Episode 100	Average Score: -0.00	Score: -0.00
Episode 200	Average Score: -0.00	Score: -0.00
Episode 300	Average Score: 0.01	Score: -0.000
Episode 400	Average Score: 0.01	Score: -0.00
Episode 500	Average Score: 0.02	Score: 0.050
Episode 600	Average Score: 0.01	Score: 0.050
Episode 700	Average Score: 0.00	Score: -0.00
Episode 800	Average Score: 0.02	Score: 0.050
Episode 900	Average Score: 0.04	Score: 0.050
Episode 1000	Average Score: 0.09	Score: 0.05
Episode 1100	Average Score: 0.06	Score: 0.100
Episode 1200	Average Score: 0.11	Score: 0.20
Episode 1300	Average Score: 0.19	Score: 0.40
Episode 1375	Average Score: 0.50	Score: 2.10
Environment solved in 1375 episodes!	Average Score: 0.50
```

During the different training trials the `maximum number of steps per episode` was enlarged up to 800 steps.

A plot of the average score per episode is illustrated here:

![Training result][image1]

### Evaluation

The evaluation of the agent for a couple of episodes.

 * [3 episodes evaluation](extra/cc01.mp4):

![Training evaluation][image2]

## Discussion and Further Work

A basic `Collaboration` and `Competition` agent was implemented to solve the task using an initial implementation from the `DDPG` algorithm. More experimentation would be useful on the hyperparameters to get better results and analyze the ammount of episodes needed. 

Other options are to radically change the architecture and algorithms to other opciones such as `PPO`, `Rainbow`, `TD3`, `MMADPG`.

Modify the selected algorithm and use `prioritied experience replay` to accelerate and stabilize training.

## Resources

* [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990)
* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)
* [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [Distributed Distributional Deep Deterministic Policy Gradient](https://openreview.net/pdf?id=SyZipzbCb)
* [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
* [TD3: Twin Delayed DDPG](https://arxiv.org/abs/1802.09477)
