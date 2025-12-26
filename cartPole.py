import random
import gymnasium as gym
import numpy as np
import time
import psutil
import os
import math

# --- CONFIGURATION ---
# Must match the Parallel version for a fair race!
NUM_EPISODES = 5000     
MAX_STEPS = 200         
WORKLOAD_DELAY = 0.002  # The "Heavy" simulation

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Discretization Buckets (Same as Parallel)
BUCKETS = (6, 12, 6, 12)

# --- HELPERS ---
def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_discrete_state_idx(state, lower_bounds, upper_bounds):
    """Converts continuous physics state to a single integer index."""
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_obs = [int(round((BUCKETS[i] - 1) * ratios[i])) for i in range(len(state))]
    new_obs = [min(BUCKETS[i] - 1, max(0, new_obs[i])) for i in range(len(state))]
    
    idx = (new_obs[0] * BUCKETS[1] * BUCKETS[2] * BUCKETS[3] + 
           new_obs[1] * BUCKETS[2] * BUCKETS[3] + 
           new_obs[2] * BUCKETS[3] + 
           new_obs[3])
    return int(idx)

# --- MAIN SETUP ---
start_time = time.time()
start_memory = get_process_memory()

env = gym.make('CartPole-v1')

# Define Bounds
upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]

# Initialize Q-Table manually (Total States x Actions)
total_states = BUCKETS[0] * BUCKETS[1] * BUCKETS[2] * BUCKETS[3]
q_table = np.zeros((total_states, env.action_space.n))

print('Training started...')

for episode in range(NUM_EPISODES):
    raw_state, info = env.reset()
    state = get_discrete_state_idx(raw_state, lower_bounds, upper_bounds)
    done = False

    for step in range(MAX_STEPS):
        # --- WORKLOAD SIMULATION ---
        time.sleep(WORKLOAD_DELAY) 
        # ---------------------------

        if random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        next_raw_state, reward, done, truncated, info = env.step(action)
        
        # Convert next state to index
        next_state = get_discrete_state_idx(next_raw_state, lower_bounds, upper_bounds)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])

        q_table[state, action] = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)

        state = next_state

        if done or truncated:
            break

    EPSILON = max(MIN_EPSILON, EPSILON_DECAY * EPSILON)

print('Training finished.\n')

# --- METRICS ---
end_time = time.time()
end_memory = get_process_memory()
execution_time = end_time - start_time

print("\n" + "="*45)
print("       PERFORMANCE EVALUATION (SINGLE CORE) ")
print("="*40)
print(f"1. Execution Time:      {execution_time:.4f} seconds")
print(f"2. Memory (RAM) Cost:   {end_memory:.2f} MB")
print(f"3. Processors Used:     1 (Single-Core)")
print("="*40 + "\n")

# --- VISUALIZATION ---
env = gym.make('CartPole-v1', render_mode='human')
for episode in range(5):
    raw_state, info = env.reset()
    state = get_discrete_state_idx(raw_state, lower_bounds, upper_bounds)
    done = False
    print('Episode:', episode)
    for step in range(200):
        env.render()
        action = np.argmax(q_table[state, :])
        next_raw_state, reward, done, truncated, info = env.step(action)
        state = get_discrete_state_idx(next_raw_state, lower_bounds, upper_bounds)
        if done or truncated:
            break
env.close()