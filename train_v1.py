# ------------------------------
# AI for play SuperMario
# Author: Linln helped by ChatGPT
# Created at 2023-4-30 2:12 am
# ------------------------------

import retro
import numpy as np

# 创建 Super Mario Bros 游戏环境
env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')

# 获取游戏屏幕的宽度和高度
width, height, _ = env.observation_space.shape

# 构建智能体


class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def get_action(self, observation):
        # 将游戏环境的观测值转换为游戏屏幕的像素值
        screen = np.array(observation)
        # 转换为灰度图像
        screen = np.mean(screen, axis=2)
        # 缩放像素值到 [0, 1] 范围内
        screen = screen.astype('float32') / 255.0
        # 展平数组为一维向量
        features = screen.flatten()
        # TODO: 在此处实现智能体的策略，并返回对应的动作
        action = self.action_space.sample()
        return action


# 创建智能体
agent = Agent(env.observation_space, env.action_space)

# 重置游戏环境
observation = env.reset()

# 运行游戏
done = False
while not done:
    # 渲染游戏屏幕
    env.render()
    # 获取智能体的动作
    action = agent.get_action(observation)
    # 在游戏环境中执行动作，并获取新的观测值、奖励、游戏是否结束的信息
    observation, reward, done, info = env.step(action)

# 关闭游戏环境
env.close()
