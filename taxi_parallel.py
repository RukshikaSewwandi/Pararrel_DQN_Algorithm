import random
import gymnasium as gym
import numpy as np
import time
import psutil
import os
import multiprocessing as mp
import ctypes

# --- CONFIGURATION ---
NUM_WORKERS = 4          # Data Parallelism: 4 Workers
NUM_EPISODES = 10000     # Total episodes
EPISODES_PER_WORKER = NUM_EPISODES // NUM_WORKERS
MAX_STEPS = 100

# Hyperparameters
ALPHA = 0.9
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01

# --- SHARED HELPERS ---

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def to_numpy_array(shared_array, shape):
    """Helper to treat shared memory array as numpy array"""
    return np.frombuffer(shared_array.get_obj()).reshape(shape)

# --- WORKER PROCESS ---
def worker_process(worker_id, shared_q_array, experience_queue, q_shape):
    """
    Worker: Owns an environment copy. 
    1. Reads Global Q-Table (shared_q_array) to act.
    2. Sends transitions to the Learner (experience_queue).
    """
    env = gym.make('Taxi-v3')
    
    # Map shared memory to numpy for easy reading
    q_table = to_numpy_array(shared_q_array, q_shape)
    
    epsilon = EPSILON_START
    
    for episode in range(EPISODES_PER_WORKER):
        state, info = env.reset()
        done = False
        
        for step in range(MAX_STEPS):
            # Choose Action (Epsilon-Greedy using Global Q-Table)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                # Read directly from shared global table
                action = np.argmax(q_table[state, :])

            next_state, reward, done, truncated, info = env.step(action)
            
            # Send Data to Global Buffer (Learner)
            # Tuple: (state, action, reward, next_state, done/truncated)
            experience_queue.put((state, action, reward, next_state, done or truncated))
            
            state = next_state
            
            if done or truncated:
                break
        
        # Local decay for exploration
        epsilon = max(MIN_EPSILON, EPSILON_DECAY * epsilon)
    
    env.close()

# --- LEARNER PROCESS ---
def learner_process(shared_q_array, experience_queue, q_shape, total_updates_needed):
    """
    Learner: 
    1. Pulls batch/transitions from Global Buffer (Queue).
    2. Updates the Global Q-Table (Shared Memory).
    """
    q_table = to_numpy_array(shared_q_array, q_shape)
    updates_count = 0
    
    # Keep learning until we process all expected steps from workers
    # Note: In a real infinite scenario, this would run forever. 
    # Here we stop when workers are likely done (simplified for script).
    while True:
        try:
            # Pull from Global Buffer
            # Timeout allows checking for exit condition if queue is empty
            state, action, reward, next_state, is_terminal = experience_queue.get(timeout=1)
            
            # --- BELLMAN UPDATE ---
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            
            # Update Global Network (Q-Table)
            q_table[state, action] = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
            
        except:
            # If queue is empty and we assume workers are done (controlled by main process usually)
            # For this script, we use a poison pill or external termination, 
            # but to match the snippet style, we break if the main process kills us.
            continue

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Environment Specs
    dummy_env = gym.make('Taxi-v3')
    obs_n = dummy_env.observation_space.n
    act_n = dummy_env.action_space.n
    q_shape = (obs_n, act_n)
    dummy_env.close()

    print(f"Training started with {NUM_WORKERS} workers...")
    
    # 2. Setup Shared Memory (The Global Network)
    shared_q_base = mp.Array(ctypes.c_double, int(obs_n * act_n))
    
    # 3. Setup Global Buffer
    experience_queue = mp.Queue()

    # --- METRICS START ---
    start_time = time.time()
    start_memory = get_process_memory()

    # 4. Start Learner
    # We run learner as a daemon so it dies when main script ends
    learner = mp.Process(target=learner_process, args=(shared_q_base, experience_queue, q_shape, NUM_EPISODES))
    learner.daemon = True
    learner.start()

    # 5. Start Workers
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(i, shared_q_base, experience_queue, q_shape))
        workers.append(p)
        p.start()

    # 6. Wait for Workers to Finish
    for p in workers:
        p.join()
    
    # Wait a moment for Learner to empty queue (flush remaining updates)
    time.sleep(1) 
    
    print('Training finished.\n')

    # --- METRICS CALCULATION ---
    end_time = time.time()
    end_memory = get_process_memory()
    execution_time = end_time - start_time
    
    # Note: Memory cost in MP is complex (shared + copy-on-write). 
    # This measures the Main process change.
    memory_cost = end_memory - start_memory 

    process = psutil.Process(os.getpid())
    num_threads = process.num_threads()

    print("\n" + "="*45)
    print("       PERFORMANCE EVALUATION       ")
    print("="*40)
    print(f"1. Execution Time:      {execution_time:.4f} seconds")
    print(f"2. Memory (RAM) Cost:   {end_memory:.2f} MB (Main Process)")
    print(f"3. Processors Used:     {NUM_WORKERS} Workers + 1 Learner")
    print(f"   - Active Threads:    {num_threads}")
    print("="*40 + "\n")

    # --- EVALUATION (VISUALIZATION) ---
    # We use the populated shared Q-table for the demo
    final_q_table = to_numpy_array(shared_q_base, q_shape)
    
    env = gym.make('Taxi-v3', render_mode='human')

    for episode in range(5):
        state, info = env.reset()
        done = False

        print('Episode:', episode)

        for step in range(MAX_STEPS):
            env.render()
            action = np.argmax(final_q_table[state, :])
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state

            if done or truncated:
                env.render()
                print('Finished episode', episode, 'with reward', reward)
                break # Added break to stop render loop on finish

    env.close()