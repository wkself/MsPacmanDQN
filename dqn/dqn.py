import random
import numpy as np
import torch
import torch.nn as nn #graphs
import torch.optim as optim #further optimization algorithms
import torch.nn.functional as F

class DQN(nn.Module):
  def __init__(self, input_shape, num_actions):
    super(DQN, self).__init__()
    #applies linear transformation to the input data
    #we define a neural network of 3 fully connected (fc) layers
    #input shape is the shape of our input state
    #num_actions is the variable representing total number of actions in env
    #in our case, num_actions is set to discrete 9 per this Atari game env
    self.fc1 = nn.Linear(input_shape, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, num_actions)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x
