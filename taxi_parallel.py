import random
import gymnasium as gym
import numpy as np
import time
import psutil
import os
import multiprocessing as mp
import ctypes

# --- CONFIGURATION ---
NUM_WORKERS = 4          
NUM_EPISODES = 20000     
EPISODES_PER_WORKER = NUM_EPISODES // NUM_WORKERS
MAX_STEPS = 100

# BATCHING CONFIG
BATCH_SIZE = 50 

# Hyperparameters
ALPHA = 0.9
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01

# --- SHARED HELPERS ---

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) 

def to_numpy_array(shared_array, shape):
    return np.frombuffer(shared_array.get_obj()).reshape(shape)

# --- WORKER PROCESS ---
def worker_process(worker_id, shared_q_array, experience_queue, q_shape):
    env = gym.make('Taxi-v3')
    q_table = to_numpy_array(shared_q_array, q_shape)
    epsilon = EPSILON_START

    local_batch = [] # Buffer
    
    for episode in range(EPISODES_PER_WORKER):
        state, info = env.reset()
        done = False
        
        for step in range(MAX_STEPS):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, truncated, info = env.step(action)
            
            # Add to local batch
            local_batch.append((state, action, reward, next_state, done or truncated))
            
            # SEND BATCH IF FULL
            if len(local_batch) >= BATCH_SIZE:
                experience_queue.put(local_batch)
                local_batch = [] 
            
            state = next_state
            if done or truncated:
                break
        
        # Send remaining data at end of episode
        if local_batch:
            experience_queue.put(local_batch)
            local_batch = []

        epsilon = max(MIN_EPSILON, EPSILON_DECAY * epsilon)
    
    env.close()

# --- LEARNER PROCESS ---
def learner_process(shared_q_array, experience_queue, q_shape, total_updates_needed):
    q_table = to_numpy_array(shared_q_array, q_shape)
    
    while True:
        try:
            # Get the list (Batch)
            batch = experience_queue.get(timeout=1) 
            
            # Iterate through the batch to update
            for transition in batch:
                state, action, reward, next_state, is_terminal = transition
                
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state, :])
                q_table[state, action] = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
            
        except:
            continue

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    dummy_env = gym.make('Taxi-v3')
    obs_n = dummy_env.observation_space.n
    act_n = dummy_env.action_space.n
    q_shape = (obs_n, act_n)
    dummy_env.close()

    print(f"Training started with {NUM_WORKERS} workers (Batching Enabled)...")
    
    shared_q_base = mp.Array(ctypes.c_double, int(obs_n * act_n))
    experience_queue = mp.Queue()

    start_time = time.time()
    start_memory = get_process_memory()

    learner = mp.Process(target=learner_process, args=(shared_q_base, experience_queue, q_shape, NUM_EPISODES))
    learner.daemon = True
    learner.start()

    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(i, shared_q_base, experience_queue, q_shape))
        workers.append(p)
        p.start()

    for p in workers:
        p.join()
    
    time.sleep(1) 
    
    print('Training finished.\n')

    end_time = time.time()
    end_memory = get_process_memory()
    execution_time = end_time - start_time
    memory_cost = end_memory - start_memory 

    process = psutil.Process(os.getpid())
    num_threads = process.num_threads()

    print("\n" + "="*45)
    print("       PERFORMANCE EVALUATION       ")
    print("="*40)
    print(f"1. Execution Time:      {execution_time:.4f} seconds")
    print(f"2. Memory (RAM) Cost:   {end_memory:.2f} MB")
    print(f"3. Processors Used:     {NUM_WORKERS} Workers + 1 Learner")
    print(f"   - Active Threads:    {num_threads}")
    print("="*40 + "\n")

    # EVALUATION
    final_q_table = to_numpy_array(shared_q_base, q_shape)
    env = gym.make('Taxi-v3', render_mode='human')

    for episode in range(5):
        state, info = env.reset()
        print('Episode:', episode)
        for step in range(MAX_STEPS):
            env.render()
            action = np.argmax(final_q_table[state, :])
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            if done or truncated:
                env.render()
                print('Finished episode', episode, 'with reward', reward)
                break 

    env.close()