import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from dqn_agent import DQNAgent

# 设置环境
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state_dim = env.observation_space.shape
action_dim = env.action_space.n

# 实例化代理
agent = DQNAgent(state_dim, action_dim)

# 加载模型
agent.load("./save/mario-dqn.h5")


def test(agent, env):
    total_reward = 0
    state = env.reset()
    state = np.reshape(state, [1, state_dim[0], state_dim[1], state_dim[2]])

    while True:
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(
            next_state, [1, state_dim[0], state_dim[1], state_dim[2]])

        state = next_state
        total_reward += reward

        if done:
            print("Score: ", total_reward)
            break

    env.close()


# 在环境中测试模型
test(agent, env)
