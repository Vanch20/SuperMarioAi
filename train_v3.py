# ------------------------------
# AI for play SuperMario V3
# Author: Linln helped by ChatGPT4
# Created at 2023-5-2 1:29 am
# ------------------------------

import numpy as np
from collections import deque
import time
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from agent import DQNAgent


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
    return image.reshape(84, 84, 1)


env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()
state = preprocess(state)
state_dim = 84 * 84 * 4
# state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

episodes = 1000
batch_size = 64
epsilon_decay = 0.995

# 记录每一轮的奖励
total_rewards = []

# 记录每一轮的最大x坐标
max_x_positions = []


stacked_frames = deque([np.zeros((84, 84, 1), dtype=int) for _ in range(4)], maxlen=4)


for episode in range(episodes):
    state = env.reset()
    state = preprocess(state)
    stacked_frames.append(state)
    state = np.concatenate(stacked_frames, axis=2)
    state = np.reshape(state, [1, state_dim])
    mario_x = 0
    last_mario_x = 0
    max_mario_x = 1
    same_position_threshold = 10
    start_time = time.time()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action.item())
        total_reward += reward
        # next_state, reward, done, info = env.step(action)
        # reward = reward if not done else -10

        mario_x = info["x_pos"]

        # 如果马里奥向前移动，奖励增加, 重置计时器
        if mario_x > max_mario_x:
            reward += 1.0
            start_time = time.time()
        # 如果马里奥停滞不前，奖励减少
        else:
            reward -= 1.0
            if time.time() - start_time > same_position_threshold:
                done = True

        last_mario_x = mario_x
        max_mario_x = max(max_mario_x, mario_x)

        env.render()
        next_state = preprocess(next_state)

        stacked_frames.append(next_state)
        next_state = np.concatenate(stacked_frames, axis=2)
        next_state = np.reshape(next_state, [1, state_dim])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
    agent.replay(batch_size)
    agent.update_epsilon(epsilon_decay)
    total_rewards.append(total_reward)
    max_x_positions.append(max_mario_x)

    print(
        f"Episode {episode}/{episodes} finished after {time.time() - start_time} with reward {total_reward} and max x position {max_mario_x}"
    )

    if episode % 10 == 0:
        agent.save("./save")
