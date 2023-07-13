import random
import numpy as np
import torch
import torch.nn as nn #graphs
import torch.optim as optim #further optimization algorithms
import torch.nn.functional as F

class ReplayBuffer():
  def __init__(self, capacity):
    self.capacity = capacity #buffer has a fixed capacity
    self.buffer = []
    self.position = 0

  def push(self, state, action, reward, next_state, done): #adds to buffer
    if len(self.buffer) < self.capacity:
      self.buffer.append(None) #overwrites the oldest experience if full
    self.buffer[self.position] = (state, action, reward, next_state, done)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size): #retrieves ran batch of experiences from buffer
    batch = random.sample(self.buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return np.array(states), actions, rewards, np.array(next_states), dones

  def __len__(self):
    return len(self.buffer) #returns number of experiences in buffer
