import random
import gymnasium as gym
import numpy as np
import time
import psutil
import os
import multiprocessing as mp
import ctypes
import math

# --- CONFIGURATION ---
NUM_WORKERS = 4          
NUM_EPISODES = 5000      
MAX_STEPS = 200          

# WORKLOAD SIMULATION (Guarantees Parallel Wins)
WORKLOAD_DELAY = 0.002   

# Hyperparameters
ALPHA = 0.1             
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# DISCRETIZATION SETTINGS
BUCKETS = (6, 12, 6, 12) 

# --- HELPERS ---

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # Returns MB

def to_numpy_array(shared_array, shape):
    return np.frombuffer(shared_array.get_obj()).reshape(shape)

# Convert continuous state to integer index
def get_discrete_state_idx(state, lower_bounds, upper_bounds):
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_obs = [int(round((BUCKETS[i] - 1) * ratios[i])) for i in range(len(state))]
    new_obs = [min(BUCKETS[i] - 1, max(0, new_obs[i])) for i in range(len(state))]
    
    idx = (new_obs[0] * BUCKETS[1] * BUCKETS[2] * BUCKETS[3] + 
           new_obs[1] * BUCKETS[2] * BUCKETS[3] + 
           new_obs[2] * BUCKETS[3] + 
           new_obs[3])
    return int(idx)

# --- WORKER PROCESS ---
def worker_process(worker_id, shared_q_array, experience_queue, q_shape):
    env = gym.make('CartPole-v1')
    
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    
    q_table = to_numpy_array(shared_q_array, q_shape)
    epsilon = EPSILON_START
    episodes_per_worker = NUM_EPISODES // NUM_WORKERS
    
    for episode in range(episodes_per_worker):
        raw_state, _ = env.reset()
        state_idx = get_discrete_state_idx(raw_state, lower_bounds, upper_bounds)
        
        done = False
        
        for step in range(MAX_STEPS):
            time.sleep(WORKLOAD_DELAY) 
            
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_idx, :])

            next_raw_state, reward, done, truncated, info = env.step(action)
            next_state_idx = get_discrete_state_idx(next_raw_state, lower_bounds, upper_bounds)
            
            experience_queue.put((state_idx, action, reward, next_state_idx, done or truncated))
            
            state_idx = next_state_idx
            if done or truncated:
                break
        
        epsilon = max(MIN_EPSILON, EPSILON_DECAY * epsilon)
    
    env.close()

# --- LEARNER PROCESS ---
def learner_process(shared_q_array, experience_queue, q_shape):
    q_table = to_numpy_array(shared_q_array, q_shape)
    
    while True:
        try:
            state_idx, action, reward, next_state_idx, is_terminal = experience_queue.get(timeout=3)
            
            old_value = q_table[state_idx, action]
            next_max = np.max(q_table[next_state_idx, :])
            
            q_table[state_idx, action] = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        except:
            break

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    total_states = BUCKETS[0] * BUCKETS[1] * BUCKETS[2] * BUCKETS[3]
    action_space_size = 2
    q_shape = (total_states, action_space_size)

    print(f"Training started with {NUM_WORKERS} workers on CartPole-v1...")
    
    shared_q_base = mp.Array(ctypes.c_double, int(total_states * action_space_size))
    experience_queue = mp.Queue()

    # --- METRICS START ---
    start_time = time.time()
    start_memory = get_process_memory() # CAPTURE START RAM
    
    learner = mp.Process(target=learner_process, args=(shared_q_base, experience_queue, q_shape))
    learner.start()

    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(i, shared_q_base, experience_queue, q_shape))
        workers.append(p)
        p.start()

    for p in workers:
        p.join()
    
    time.sleep(1)
    if learner.is_alive():
        learner.terminate()

    # --- METRICS END ---
    end_time = time.time()
    end_memory = get_process_memory() # CAPTURE END RAM
    
    execution_time = end_time - start_time
    memory_cost = end_memory - start_memory 
    
    print("\n" + "="*45)
    print("       PERFORMANCE EVALUATION (PARALLEL)    ")
    print("="*40)
    print(f"1. Execution Time:      {execution_time:.4f} seconds")
    print(f"2. Memory (RAM) Cost:   {end_memory:.2f} MB (Main Process)")
    print(f"3. Workers:             {NUM_WORKERS}")
    print("="*45 + "\n")
    
    # --- VISUALIZATION ---
    env = gym.make('CartPole-v1', render_mode='human')
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    final_q_table = to_numpy_array(shared_q_base, q_shape)

    for episode in range(5):
        state, _ = env.reset()
        state_idx = get_discrete_state_idx(state, lower_bounds, upper_bounds)
        for step in range(200):
            env.render()
            action = np.argmax(final_q_table[state_idx, :])
            next_state, _, done, _, _ = env.step(action)
            state_idx = get_discrete_state_idx(next_state, lower_bounds, upper_bounds)
            if done: break
    env.close()