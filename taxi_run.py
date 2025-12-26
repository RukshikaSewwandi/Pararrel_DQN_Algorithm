import random
import gymnasium as gym
import numpy as np

import time
import psutil
import os

# --- MONITORING SETUP ---
def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert Bytes to MB

# Start the Timer and Resource Monitor
start_time = time.time()
start_memory = get_process_memory()


env = gym.make('Taxi-v3') # Create the Taxi environment
# env = gym.make('Taxi-v3', render_mode='human') #Uncomment to visualize the environment

alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 20000
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

        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])

        q_table[state, action] = (1 - alpha) *old_value + alpha * (reward + gamma * next_max)

        state = next_state

        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon_decay * epsilon)

print('Training finished.\n')


# --- METRICS CALCULATION ---
end_time = time.time()
end_memory = get_process_memory()
execution_time = end_time - start_time
memory_cost = end_memory - start_memory


# Get CPU info
process = psutil.Process(os.getpid())
cpu_usage = process.cpu_percent(interval=None) / psutil.cpu_count() # Normalized by cores
num_threads = process.num_threads()

print("\n" + "="*45)
print("       PERFORMANCE EVALUATION       ")
print("="*40)
print(f"1. Execution Time:      {execution_time:.4f} seconds")
print(f"2. Memory (RAM) Cost:   {end_memory:.2f} MB (Peak usage)")
print(f"3. Processors Used:     1 (Single-Core)")
print(f"   - Active Threads:    {num_threads}")
print("="*40 + "\n")


env = gym.make('Taxi-v3', render_mode='human')

for episode in range(5):
    state, info = env.reset()
    done = False

    print('Episode:', episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        if done or truncated:
            env.render()
            print('Finished episode', episode, 'with reward', reward)

env.close()