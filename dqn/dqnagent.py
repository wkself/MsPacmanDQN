import random
import numpy as np
import torch
import torch.nn as nn #graphs
import torch.optim as optim #further optimization algorithms
import torch.nn.functional as F

class DQNAgent():
  def __init__(self, env, replay_buffer, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    #our initialization method sets up our networks, optimizer, and loss func
    self.env = env
    self.replay_buffer = replay_buffer
    self.batch_size = batch_size
    self.gamma = gamma
    self.epsilon = epsilon_start
    self.epsilon_end = epsilon_end
    self.epsilon_decay = epsilon_decay

    self.policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
    self.target_net = DQN(env.observation_space.shape[0], env.action_space.n)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer.optim.Adam(self.policy_net.parameters())
    self.loss_fn = nn.SmoothL1Loss()

  def select_action(self, state):
    #selects action based on epsilon-greedy policy
    if random.random() < self.epsilon:
      return self.env.action_space.sample()
    else:
      with torch.no_grad():
        q_values = self.policy_net(torch.tensor(state, dtype=torch.float32))
        return q_values.argmax().item()

  def train(self, num_episodes):
    #runs training loop, we specify the number of episodes
    #for every episode, agent interacts with environment, collects experiences,
    #and updates the networks
    episode_rewards = []
    for episode in range(num_episodes):
      state = self.env.reset()
      done = False
      total_reward = 0

      while not done:
        action = self.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        self.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(self.replay_buffer) >= self.batch_size:
          states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
          self.update_network(states, actions, rewards, next_states, dones)

      self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
      episode_rewards.append(total_reward)

      if episode % 100 == 0:
        print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:])}")

  def update_network(self, states, actions, rewards, next_states, dones):
    #calculates loss, performs gradient descent, updates policy network
    #updates target network periodically
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = self.policy_net(states).gather(1, actions)
    next_q_values = self.target_net(next_states).max(1)[0].detach()
    expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

    loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.update_target_network()

  def update_target_network(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())
