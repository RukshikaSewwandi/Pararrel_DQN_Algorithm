import random
import gymnasium as gym
import numpy as np

env = gym.make('Taxi-v3') # Create the Taxi environment
# env = gym.make('Taxi-v3', render_mode='human') #Uncomment to visualize the environment

alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100


q_table = np.zeros((env.observation_space.n, env.action_space.n))


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

print('Training started...')

for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    for step in range(max_steps):
        #env.render() # Uncomment to visualize training
        action = choose_action(state)

        next_state, reward, done, turncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])

        q_table[state, action] = (1 - alpha) *old_value + alpha * (reward + gamma * next_max)

        state = next_state

        if done or turncated:
            break

    epsilon = max(min_epsilon, epsilon_decay * epsilon)


print('Training ended...')

env = gym.make('Taxi-v3', render_mode='human')

for episode in range(5):
    state, info = env.reset()
    done = False

    print('Episode:', episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, done, turncated, info = env.step(action)
        state = next_state

        if done or turncated:
            env.render()
            print('Finished episode', episode, 'with reward', reward)

env.close()