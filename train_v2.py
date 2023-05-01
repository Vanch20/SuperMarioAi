# ------------------------------
# AI for play SuperMario V2
# Author: Linln helped by ChatGPT
# ------------------------------

import retro
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from collections import deque

# 创建游戏环境
env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')

# 设置随机数种子
np.random.seed(123)
# tf.random.set_seed(123)

# 定义模型


def create_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
              input_shape=(240, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(env.action_space.n))
    return model


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state, done = zip(*batch)
        # return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
        return batch


# 创建模型
model = create_model()

# 编译模型
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse')

# 定义参数
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = ReplayBuffer(10000)
max_memory_length = 100000
num_episodes = 10000
timesteps_per_episode = 10000
update_target_network_every = 1000
target_network = create_model()
target_network.set_weights(model.get_weights())
save_freq = 100

# 训练模型
for episode in range(num_episodes):
    state = env.reset()
    state = cv2.resize(state, (256, 240))
    state = np.array(state)
    state = np.expand_dims(state, axis=0)
    done = False
    total_reward = 0
    steps = 0
    while not done:
        # print('state shape:', np.array(state).shape)
        # Choose epsilon-greedy action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state.reshape(1, *state.shape)))
        next_state, reward, done, info = env.step(action)

        next_state = np.array(next_state)
        next_state = np.expand_dims(next_state, axis=0)
        total_reward += reward

        # Update memory
        memory.push(state, action, reward, next_state, done)

        # print('nextstate shape:', np.array(next_state).shape)
        state = next_state
        steps += 1

        # Update target network weights
        if steps % update_target_network_every == 0:
            target_network.set_weights(model.get_weights())

        # Render screen
        env.render()

        # Train model if memory is sufficient
        # if len(memory) >= batch_size:
        #     experiences = memory.sample(batch_size)
        #     states, actions, rewards, next_states, dones = zip(*experiences)
        #     print('rewards:', np.array(rewards))
        #     # states = states
        #     # actions = (actions)
        #     # rewards = (rewards)
        #     # next_states = (next_states)
        #     # dones = (dones)
        #     targets = rewards + (1 - dones[0]) * gamma * \
        #         np.amax(target_network.predict(next_states), axis=1)
        #     targets_full = model.predict(states)
        #     ind = np.array([i for i in range(batch_size)])
        #     targets_full[[ind], [actions]] = targets
        #     history = model.fit(states, targets_full,
        #                         epochs=num_episodes, verbose=0)

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print episode results
    print(
        f"Episode {episode + 1}: total reward -> {total_reward}, epsilon -> {epsilon:.4f}, steps -> {steps}")

    # Save model every save_freq episodes
    if (episode + 1) % save_freq == 0:
        model.save_weights(f"mario_model_episode_{episode + 1}.h5")

# Close environment
env.close()
