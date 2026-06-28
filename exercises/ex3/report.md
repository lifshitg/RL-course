# Programming Assignment 3: RL Algorithms Comparison

## 1. Project Introduction
This project compares the empirical performance and learning dynamics of three fundamental Reinforcement Learning algorithms: Tabular Q-learning, Tabular SARSA, and Stochastic Tabular REINFORCE. 

The selected environment is Gymnasium's `FrozenLake-v1` on an 8x8 grid configured with stochastic dynamics (`is_slippery=True`). This introduces severe noise into the Markov Decision Process (MDP), as intended actions succeed only 33% of the time, while the remaining 66% result in orthogonal slips. Furthermore, the environment features a sparse reward structure, where only the single goal state provides a reward of +1, and all other states (including terminal holes) provide 0.

### Included Files & Execution Instructions
* `algorithms.py`: Contains the core algorithm implementations, the isolated evaluation loop, and the graph plotting logic.
* `.env`: A configuration file holding all hyperparameters, parsed automatically by the script to allow easy tuning without modifying the Python code.
* `result.png`: The final generated plot comparing the empirical performance of Q-learning, SARSA, and REINFORCE using the tuned parameters.
* `log.txt`: A detailed execution log capturing the step-by-step $V(s_0)$ evaluation scores for all algorithms across all random seeds.
* `bonus_original_parameters.png`: An additional plot demonstrating the algorithms' performance using the initial baseline parameters, provided for empirical comparison against the tuned results.

**How to Run:**
1. Ensure you have the required libraries installed in your Python environment:
   `pip install gymnasium numpy matplotlib`
2. Place `algorithms.py` and `.env` in the same directory.
3. Execute the script from your terminal:
   `python algorithms.py`
4. The script will output training progress to the console (and `log.txt`) and save the final comparison plot as `result.png`.

---

## 2. Hyperparameters and Tuning

### General & Baseline Parameters
* **`ENV_NAME=FrozenLake-v1`**: The target environment.
* **`IS_SLIPPERY=True`**: Enables the highly stochastic slip dynamics.
* **`MAX_STEPS_PER_EPISODE=200`**: Truncation limit to prevent infinite loops when the agent gets stuck sliding on the ice.
* **`TOTAL_TRAINING_STEPS=500000`**: The global stopping condition for training.
* **`EVAL_INTERVAL_STEPS=10000`**: The frequency at which the training loop pauses to run the isolated evaluation.
* **`N_EVAL_EPISODES=500`**: The number of independent episodes run per evaluation phase to calculate a statistically significant mean for $V(s_0)$.
* **`GAMMA=0.99`**: The discount factor. It forces the agent to find the shortest possible path, as delayed rewards lose their value over time.
* **`THETA_CLIP=20.0`**: Used in REINFORCE to clamp the policy preferences ($h(s,a)$). This prevents numerical overflow (`NaN`s) or underflow when calculating the exponential function within the Softmax policy.

### Omitted Parameters
* **`use_baseline=False` & `value_lr`**: The assignment explicitly required running REINFORCE without a value baseline. Because no baseline value function $V(s)$ is maintained or updated during training, the `value_lr` parameter is obsolete and was strictly omitted from both the `.env` configuration and the codebase.

### Exploration Parameters (Q-learning & SARSA)
* **`EPSILON_START=1.0`**, **`EPSILON_END=0.05`**, **`EPSILON_DECAY_STEPS=300000`**
  * *Role:* Controls the $\epsilon$-greedy exploration rate. 
  * *Justification:* A highly extended decay (over 60% of total training steps) is absolutely critical here. Due to the extreme sparse reward and slippery grid, a purely random agent is highly unlikely to stumble upon the goal. If exploration ends too early, the agent's $Q$-table remains populated entirely with zeros, causing it to greedily walk into walls or holes indefinitely, yielding a flat $0.0$ return.

### Tuned Parameters and Theoretical Justifications
To achieve meaningful convergence, specific parameters were tuned away from the assignment's initial baselines:

* **`ALPHA=0.05`** (Changed from baseline `0.1`)
  * *Theoretical:* In value-based methods, $\alpha$ dictates how heavily new experiences overwrite existing knowledge. In a stochastic environment like slippery FrozenLake, transitions are incredibly noisy. A high learning rate makes the $Q$-values overly sensitive to random, unlucky slips (e.g., slipping into a hole from a historically safe state), destroying learned policies.
  * *Empirical:* In tests with the original $\alpha=0.1$ (as seen in `bonus_original_parameters.png`), Q-learning reached an early peak but subsequently suffered violent performance crashes, oscillating wildly. Lowering $\alpha$ to 0.05 acted as a low-pass filter, effectively smoothing out the stochastic noise. It allowed SARSA to cleanly converge and prevented Q-learning's late-stage catastrophic collapse.

* **`POLICY_LR=0.01`** (Changed from baseline `0.005`)
  * *Theoretical:* Pure Monte Carlo REINFORCE without a baseline suffers from massive variance. Because rewards are sparse, the vast majority of episodes yield a return of 0. When a rare successful trajectory (+1) is found, a conservative learning rate like 0.005 fails to update the policy probabilities strongly enough to encourage repeating that trajectory.
  * *Empirical:* With the original `0.005` baseline, the REINFORCE learning curve completely flatlined near $0.0$. Doubling it to `0.01` gave the gradients enough magnitude to meaningfully shift the softmax distributions, allowing the algorithm to finally exhibit an upward learning trend.

---

## 3. Evaluation of V(s0)

The empirical value of the initial state, $V(s_0)$, is calculated strictly outside the training loop to preserve the training sequence:
1. Training is paused exactly every 10,000 environment steps.
2. A separate, isolated evaluation environment is initialized.
3. The current policy is frozen (no updates occur to the $Q$-table or $\theta$).
4. The agent runs 500 episodes starting from the initial state $s_0$.
5. Action selection is purely greedy: $a = \arg\max Q(s, \cdot)$ for value-based methods, and $a = \arg\max \pi_\theta(\cdot|s)$ for REINFORCE.
6. The discounted return $G = \sum \gamma^t R_{t+1}$ is calculated for each episode.
7. $V(s_0)$ is reported as the arithmetic mean of these 500 returns.

---

## 4. Discussion of Results

Based on the empirical data visualized in the tuned graph (`result.png`), we can draw decisive conclusions regarding the five key aspects of the algorithms' learning dynamics:

* **Who learned faster?**
  Empirically, **Q-learning** learned the fastest. The blue curve shows the earliest and sharpest rise in performance, breaking away from $0.0$ around step 50,000 and dominating the first 200,000 steps. Theoretically, as an off-policy algorithm, Q-learning updates its estimates based on the maximum possible future reward ($\max Q$). This aggressive bootstrapping propagates the sparse reward signal back to the start state much quicker than on-policy methods.

* **Who achieved better performance?**
  In this tuned run ($\alpha=0.05$), **Q-learning** achieved the best overall expected return, hovering around an average $V(s_0)$ of $0.10 - 0.12$ in the latter half of training. **SARSA** followed closely behind, stabilizing around $0.08 - 0.10$. REINFORCE performed the worst, barely reaching $0.04$. The lower learning rate successfully prevented Q-learning from experiencing the catastrophic collapses typical of off-policy methods in noisy environments, allowing it to maintain its performance lead.

* **Did Q-learning and SARSA behave differently?**
  Yes, their behavioral differences are highly visible. Q-learning (blue) exhibits a more jagged and volatile curve with higher peaks and steeper drops. Because it assumes optimal future actions, it learns a riskier, shorter path near the holes. When it inevitably slips, it receives severe penalties, causing those sharp drops. Conversely, SARSA (orange) climbs more gradually and exhibits slightly smoother transitions. As an on-policy algorithm, SARSA factors in the $\epsilon$-greedy exploration and environmental slips, leading it to learn a "pessimistic" and safer route. This safer route takes longer to traverse, which is heavily discounted by $\gamma=0.99$, explaining why SARSA's peak expected return is slightly lower than Q-learning's.

* **Was REINFORCE less stable?**
  Empirically, no. Looking at the graph, REINFORCE (green) actually displays the *narrowest* variance across different random seeds (the tightest shaded region) and a smooth, steady upward trajectory. Visually, it appears to be the most stable algorithm. However, this empirical consistency is a direct consequence of compensating for its theoretical *instability*. Because REINFORCE without a value baseline suffers from massive variance in its Monte Carlo gradient updates, the learning rate must be severely restricted (`POLICY_LR = 0.01`) to prevent the policy from crashing. This tiny step size forces the algorithm to update its probabilities extremely slowly and cautiously. Since all random seeds face the same sparse-reward struggle and update at this glacial pace, they diverge very little from one another. This results in the narrow variance band and the steady, suppressed performance curve compared to the faster TD methods.

* **How did the stochasticity of FrozenLake affect learning?**
  The `is_slippery=True` stochasticity fundamentally shaped the learning process in two ways. First, it capped the maximum possible return. Even with a near-perfect policy, the agent will frequently slip into holes by pure chance. Therefore, $V(s_0)$ converges around $0.12$ rather than $1.0$, as most episodes inherently end in failure. Second, the stochasticity caused the massive variance (wide shaded regions) seen in the value-based algorithms. A policy might succeed in one evaluation phase and fail completely in the next purely due to the 66% slip probability, forcing the algorithms to continuously battle environmental noise to isolate the true optimal policy.