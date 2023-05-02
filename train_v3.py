# ------------------------------
# AI for play SuperMario V3
# Author: Linln helped by ChatGPT4
# Created at 2023-5-2 1:29 am
# ------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time
import cv2
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
    return image.reshape(84, 84, 1)


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
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.loss_fn = nn.MSELoss()
        self.action_space = env.action_space.n

        # 判断角色是否卡住不能前进
        self.same_position_counter = 0
        self.same_position_threshold = 5  # Set threshold
        self.last_mario_x = 0
        # self.target_model = self._build_model()
        # self.update_target_model()

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 转换为张量
        if np.random.rand() <= self.epsilon:
            return torch.tensor([random.randrange(self.action_space)], dtype=torch.long)
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
            prediction = self.model(state).gather(
                1, action.unsqueeze(1)).squeeze(1)

            loss = self.loss_fn(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self, value):
        self.epsilon *= value


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()
state = preprocess(state)
# state_dim = state.shape[0] * state.shape[1] * state.shape[2]
state_dim = 84 * 84 * 4
# state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

episodes = 1000
batch_size = 64
epsilon_decay = 0.995

stacked_frames = deque([np.zeros((84, 84, 1), dtype=int)
                       for _ in range(4)], maxlen=4)


for episode in range(episodes):
    state = env.reset()
    state = preprocess(state)
    stacked_frames.append(state)
    state = np.concatenate(stacked_frames, axis=2)
    state = np.reshape(state, [1, state_dim])
    mario_x = 0
    agent.last_mario_x = mario_x
    start_time = time.time()
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action.item())
        # next_state, reward, done, info = env.step(action)
        # reward = reward if not done else -10

        env.render()
        mario_x = info['x_pos']
        print(mario_x)
        if mario_x <= agent.last_mario_x:
            if time.time() - start_time > agent.same_position_threshold:
                done = True
        else:
            agent.last_mario_x = mario_x
            start_time = time.time()

        next_state = preprocess(next_state)

        stacked_frames.append(next_state)
        next_state = np.concatenate(stacked_frames, axis=2)
        next_state = np.reshape(next_state, [1, state_dim])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
    agent.replay(batch_size)
    agent.update_epsilon(epsilon_decay)

    if episode % 10 == 0:
        print(f"Episode {episode}/{episodes} finished.")
