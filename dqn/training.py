import random
import numpy as np
import torch
import torch.nn as nn #graphs
import torch.optim as optim #further optimization algorithms
import torch.nn.functional as F

replay_buffer = ReplayBuffer(capacity=10000)
agent = DQNAgent(env, replay_buffer, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99)
agent.train(num_episodes=1000)

#instantiate our classes, set capacities, pass variables to our agent instance

episode_rewards = []
for episode in range(100):
  state=env.reset()
  done = False
  total_reward = 0

  while not done:
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward

  episode_rewards.append(total_reward)
