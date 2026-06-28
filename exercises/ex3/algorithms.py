import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import sys

# Custom logger to simultaneously print to console and write to a log file
class Logger:
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("log.txt")

# Helper function to read the .env file safely (using utf-8-sig to bypass Windows BOM issues)
def load_env(filepath=".env"):
    print(f"[*] Loading environment variables from {filepath}...")
    env_vars = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip()
    print(f"[*] Successfully loaded {len(env_vars)} variables.")
    return env_vars

# Load and cast all the hyperparameters
config = load_env()

ENV_NAME = config["ENV_NAME"]
IS_SLIPPERY = config["IS_SLIPPERY"] == "True"
MAX_STEPS = int(config["MAX_STEPS_PER_EPISODE"])
TOTAL_STEPS = int(config["TOTAL_TRAINING_STEPS"])
EVAL_INTERVAL = int(config["EVAL_INTERVAL_STEPS"])
N_EVAL = int(config["N_EVAL_EPISODES"])

ALPHA = float(config["ALPHA"])
GAMMA = float(config["GAMMA"])
EPS_START = float(config["EPSILON_START"])
EPS_END = float(config["EPSILON_END"])
EPS_DECAY = float(config["EPSILON_DECAY_STEPS"])

POLICY_LR = float(config["POLICY_LR"])
THETA_CLIP = float(config["THETA_CLIP"])

# Evaluates the current policy purely greedily (no learning or exploration)
def evaluate_policy(q_table, theta, algo_type, n_eval_episodes, max_steps, gamma, seed):
    env = gym.make(ENV_NAME, map_name="8x8", is_slippery=IS_SLIPPERY)
    returns = []
    for i in range(n_eval_episodes):
        eval_seed = seed * 1000 + i
        state, _ = env.reset(seed=eval_seed)
        G = 0.0
        discount = 1.0
        for _ in range(max_steps):
            # Choose the best action deterministically depending on the algorithm
            if algo_type in ["q_learning", "sarsa"]:
                action = int(np.argmax(q_table[state]))
            else:
                # For REINFORCE, use softmax probabilities to pick the most likely action
                x = theta[state] - np.max(theta[state]) # Subtracted max for numerical stability
                exp_x = np.exp(x)
                probs = exp_x / np.sum(exp_x)
                action = int(np.argmax(probs))
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            G += discount * reward
            discount *= gamma
            if terminated or truncated:
                break
            state = next_state
        returns.append(G)
    env.close()
    return float(np.mean(returns))

def q_learning(seed):
    print(f"\n[Q-Learning] Starting training for seed {seed}...")
    env = gym.make(ENV_NAME, map_name="8x8", is_slippery=IS_SLIPPERY)
    env.action_space.seed(seed)
    np.random.seed(seed)
    
    # Initialize Q-table with zeros as the baseline for tabular methods
    q_table = np.zeros((64, 4))
    steps = 0
    eval_steps = []
    eval_results = []
    state, _ = env.reset(seed=seed)

    while steps < TOTAL_STEPS:
        # Periodic evaluation
        if steps % EVAL_INTERVAL == 0:
            print(f"[Q-Learning] Seed {seed} | Evaluating at step {steps}/{TOTAL_STEPS}...")
            eval_steps.append(steps)
            eval_results.append(evaluate_policy(q_table, None, "q_learning", N_EVAL, MAX_STEPS, GAMMA, seed))
            print(f"[Q-Learning] Seed {seed} | V(s0) = {eval_results[-1]:.4f}")

        # Epsilon-greedy action selection
        epsilon = max(EPS_END, EPS_START - steps / EPS_DECAY * (EPS_START - EPS_END))
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        steps += 1

        # Q-learning TD Update (Off-policy: taking the max over next possible actions)
        td_target = reward
        if not (terminated or truncated):
            td_target += GAMMA * np.max(q_table[next_state])
        q_table[state, action] += ALPHA * (td_target - q_table[state, action])

        if terminated or truncated:
            state, _ = env.reset(seed=seed)
        else:
            state = next_state

    # Final evaluation catch
    if steps % EVAL_INTERVAL == 0 and steps not in eval_steps:
        print(f"[Q-Learning] Seed {seed} | Final evaluation at step {steps}...")
        eval_steps.append(steps)
        eval_results.append(evaluate_policy(q_table, None, "q_learning", N_EVAL, MAX_STEPS, GAMMA, seed))
        print(f"[Q-Learning] Seed {seed} | Final V(s0) = {eval_results[-1]:.4f}")

    env.close()
    print(f"[Q-Learning] Finished training for seed {seed}.")
    return eval_steps, eval_results

def sarsa(seed):
    print(f"\n[SARSA] Starting training for seed {seed}...")
    env = gym.make(ENV_NAME, map_name="8x8", is_slippery=IS_SLIPPERY)
    env.action_space.seed(seed)
    np.random.seed(seed)
    q_table = np.zeros((64, 4))
    steps = 0
    eval_steps = []
    eval_results = []
    state, _ = env.reset(seed=seed)
    
    # Initial action selection
    epsilon = max(EPS_END, EPS_START - steps / EPS_DECAY * (EPS_START - EPS_END))
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(q_table[state]))

    while steps < TOTAL_STEPS:
        if steps % EVAL_INTERVAL == 0:
            print(f"[SARSA] Seed {seed} | Evaluating at step {steps}/{TOTAL_STEPS}...")
            eval_steps.append(steps)
            eval_results.append(evaluate_policy(q_table, None, "sarsa", N_EVAL, MAX_STEPS, GAMMA, seed))
            print(f"[SARSA] Seed {seed} | V(s0) = {eval_results[-1]:.4f}")

        next_state, reward, terminated, truncated, _ = env.step(action)
        steps += 1

        # Choose the *next* action before the update (On-policy requirement)
        epsilon = max(EPS_END, EPS_START - steps / EPS_DECAY * (EPS_START - EPS_END))
        if np.random.rand() < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = int(np.argmax(q_table[next_state]))

        # SARSA TD Update (On-policy: using the actual next action chosen by epsilon-greedy)
        td_target = reward
        if not (terminated or truncated):
            td_target += GAMMA * q_table[next_state, next_action]

        q_table[state, action] += ALPHA * (td_target - q_table[state, action])

        if terminated or truncated:
            state, _ = env.reset(seed=seed)
            epsilon = max(EPS_END, EPS_START - steps / EPS_DECAY * (EPS_START - EPS_END))
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
        else:
            state = next_state
            action = next_action
            
    if steps % EVAL_INTERVAL == 0 and steps not in eval_steps:
        print(f"[SARSA] Seed {seed} | Final evaluation at step {steps}...")
        eval_steps.append(steps)
        eval_results.append(evaluate_policy(q_table, None, "sarsa", N_EVAL, MAX_STEPS, GAMMA, seed))
        print(f"[SARSA] Seed {seed} | Final V(s0) = {eval_results[-1]:.4f}")

    env.close()
    print(f"[SARSA] Finished training for seed {seed}.")
    return eval_steps, eval_results

def reinforce(seed):
    print(f"\n[REINFORCE] Starting training for seed {seed}...")
    env = gym.make(ENV_NAME, map_name="8x8", is_slippery=IS_SLIPPERY)
    env.action_space.seed(seed)
    np.random.seed(seed)
    
    # Initialize policy weights (theta) to zero as requested
    theta = np.zeros((64, 4))
    steps = 0
    eval_steps = []
    eval_results = []
    state, _ = env.reset(seed=seed)
    
    # Buffers to store the trajectory for the Monte Carlo episode update
    ep_states = []
    ep_actions = []
    ep_rewards = []

    while steps < TOTAL_STEPS:
        if steps % EVAL_INTERVAL == 0:
            print(f"[REINFORCE] Seed {seed} | Evaluating at step {steps}/{TOTAL_STEPS}...")
            eval_steps.append(steps)
            eval_results.append(evaluate_policy(None, theta, "reinforce", N_EVAL, MAX_STEPS, GAMMA, seed))
            print(f"[REINFORCE] Seed {seed} | V(s0) = {eval_results[-1]:.4f}")

        # Calculate probabilities using Softmax
        x = theta[state] - np.max(theta[state]) # Stability trick to prevent overflow
        exp_x = np.exp(x)
        probs = exp_x / np.sum(exp_x)
        
        # Sample action from the calculated probability distribution
        action = np.random.choice(4, p=probs)

        next_state, reward, terminated, truncated, _ = env.step(action)
        steps += 1

        ep_states.append(state)
        ep_actions.append(action)
        ep_rewards.append(reward)

        # Once episode is done, perform the update over the entire trajectory
        if terminated or truncated or len(ep_states) >= MAX_STEPS:
            G = 0.0
            # Iterate backwards to calculate the discounted return G_t for each step
            for t in reversed(range(len(ep_rewards))):
                G = ep_rewards[t] + GAMMA * G
                st = ep_states[t]
                at = ep_actions[t]
                
                # Recalculate probabilities for the gradient
                x_t = theta[st] - np.max(theta[st])
                exp_x_t = np.exp(x_t)
                probs_t = exp_x_t / np.sum(exp_x_t)
                
                # Gradient of log-policy for softmax: 1 - P(a) for the chosen action, -P(a) for others
                grad = -probs_t
                grad[at] += 1.0
                
                # Update theta using the policy gradient theorem (without baseline)
                theta[st] += POLICY_LR * (GAMMA ** t) * G * grad
            
            # Clip theta to avoid numerical instability (NaN values)
            theta = np.clip(theta, -THETA_CLIP, THETA_CLIP)
            
            # Clear buffers for the next episode
            ep_states = []
            ep_actions = []
            ep_rewards = []
            state, _ = env.reset(seed=seed)
        else:
            state = next_state
            
    if steps % EVAL_INTERVAL == 0 and steps not in eval_steps:
        print(f"[REINFORCE] Seed {seed} | Final evaluation at step {steps}...")
        eval_steps.append(steps)
        eval_results.append(evaluate_policy(None, theta, "reinforce", N_EVAL, MAX_STEPS, GAMMA, seed))
        print(f"[REINFORCE] Seed {seed} | Final V(s0) = {eval_results[-1]:.4f}")

    env.close()
    print(f"[REINFORCE] Finished training for seed {seed}.")
    return eval_steps, eval_results

def main():
    seeds = [0, 1, 2, 3, 4] # Use 5 seeds to capture statistical significance
    
    q_results_all = []
    sarsa_results_all = []
    reinforce_results_all = []
    
    steps_arr = None

    print("\n" + "="*50)
    print("STARTING EXPERIMENTS")
    print("="*50)

    # Run all algorithms for each seed sequentially
    for s in seeds:
        print(f"\n>>> PROCESSING SEED {s} <<<")
        
        s_steps, q_res = q_learning(s)
        q_results_all.append(q_res)
        if steps_arr is None:
            steps_arr = s_steps
            
        _, sarsa_res = sarsa(s)
        sarsa_results_all.append(sarsa_res)
        
        _, reinforce_res = reinforce(s)
        reinforce_results_all.append(reinforce_res)

    print("\n" + "="*50)
    print("ALL SEEDS COMPLETED. CALCULATING STATISTICS AND PLOTTING...")
    print("="*50)

    # Calculate mean and standard deviation across all seeds
    q_mean = np.mean(q_results_all, axis=0)
    q_std = np.std(q_results_all, axis=0)
    sarsa_mean = np.mean(sarsa_results_all, axis=0)
    sarsa_std = np.std(sarsa_results_all, axis=0)
    reinforce_mean = np.mean(reinforce_results_all, axis=0)
    reinforce_std = np.std(reinforce_results_all, axis=0)

    # Plotting the learning curves with standard deviation shadows
    plt.figure(figsize=(10, 6))
    
    plt.plot(steps_arr, q_mean, label="Q-learning")
    plt.fill_between(steps_arr, q_mean - q_std, q_mean + q_std, alpha=0.2)
    
    plt.plot(steps_arr, sarsa_mean, label="SARSA")
    plt.fill_between(steps_arr, sarsa_mean - sarsa_std, sarsa_mean + sarsa_std, alpha=0.2)
    
    plt.plot(steps_arr, reinforce_mean, label="REINFORCE")
    plt.fill_between(steps_arr, reinforce_mean - reinforce_std, reinforce_mean + reinforce_std, alpha=0.2)

    plt.xlabel("Number of environment steps")
    plt.ylabel("Estimated value of initial state")
    plt.title("Comparison of Q-learning, SARSA, and REINFORCE on FrozenLake 8x8")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("result_new.png", dpi=300, bbox_inches="tight")
    print("[*] Plot saved successfully to bonus_original_parameters.png")
    
    plt.show()
    print("[*] Script execution finished successfully.")
    plt.close()

if __name__ == "__main__":
    main()