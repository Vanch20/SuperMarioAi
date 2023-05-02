import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import datetime


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    lr = 0.001

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.memory = deque(maxlen=10000)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 转换为张量
        if np.random.rand() <= self.epsilon:
            return torch.tensor([random.randrange(self.action_dim)], dtype=torch.long)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).unsqueeze(0)  # returns action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            if done:
                target = reward
            else:
                target = reward + self.gamma * \
                    torch.max(self.model(next_state))
            # prediction = self.model(state)[action]
            prediction = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)

            loss = self.loss_fn(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self, value):
        self.epsilon *= value

    def save(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        now = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        torch.save(self.model.state_dict(), "{}/model_{}.h5".format(output_dir, now))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
