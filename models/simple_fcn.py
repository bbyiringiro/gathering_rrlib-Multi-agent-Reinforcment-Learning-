import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from utils.constant import DQNSetting
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, in_features=1008, num_actions=8):
        """
        A model described in the
        https://storage.googleapis.com/deepmind-media/papers/multi-agent-rl-in-ssd.pdf
        :param in_features: number of features of input
        :param num_actions: total number of all possible actions in the game gathering and wolfpack
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, x):
        """
        the input batch of state is reshaped to (batch size, in_features = input channels * img_height * img_width)
        :param x: input shape are (batch size, input channels, img_height, img_width)
        :return:
        """
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)