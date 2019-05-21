# frozen lake with Q-Learning

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
print("First Q-Table Values")
print(Q)
# Q = np.random.normal(loc=0.1, scale=1, size=(env.observation_space.n, env.action_space.n))
dis = 0.95
r = 0.6
num_episodes = 4000
rList = []
for_graph = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    # / -> // 연산자로 바꿈
    e = 1. / ((i // 100) + 1)

    while not done:
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, done, _ = env.step(action)
        # Q[state, action] = reward + dis * np.max(Q[new_state, :])
        Q[state, action] += r * (reward + dis * np.max(Q[new_state, :]) - Q[state, action])
        rAll += reward
        state = new_state

    rList.append(rAll)
    for_graph.append((sum(rList) / (i + 1) * 100))

print("Success rate : " + str(sum(rList) / num_episodes * 100))
print("Final Q-Table Values")
print(Q)

# plt.bar(range(len(rList)), rList, color="black")
plt.plot(range(len(rList)), for_graph, linewidth=1.0, color="black")
plt.show()
