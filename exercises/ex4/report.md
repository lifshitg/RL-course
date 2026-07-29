# Programming Assignment 4: Planning under Partial Observability (POMDP) - Box Pushing with POMCP

## How to Run
To execute the primary experiment and reproduce the results, run the following command from the project root:
```bash
python .\exercises\ex4\main.py
```

## 1. Observation Function Alternative
**Choice:** Alternative B (Egocentric Observation) - the code for it is implemented within environment.py file.

**Reasoning:** 
An egocentric 3x3 window centered directly on the agent provides symmetric, localized spatial awareness. In the Box Pushing grid environment, movement actions are absolute (Up, Down, Left, Right) rather than dependent on the agent's "forward-facing" direction. Therefore, providing an orientation-independent window surrounding the agent is critical for accurately updating the Particle Filter weights. A fixed directional window (Alternative A) would leave the agent entirely blind to adjacent entities in opposing directions, unnecessarily hindering the belief state convergence. Additionally, from a practical development perspective, this symmetric approach is significantly easier to implement, as it eliminates the need to continuously track, calculate, and rotate the observation space based on the agent's current heading.
## 2. Hyperparameters
*   **n_particles:** 500 (Ensures sufficient belief state density without excessive computational overhead).
*   **c (UCB1 Exploration Constant):** 2.0 (Determined via a grid search evaluating values from 0.5 to 10.0. The value of 2.0 provided the optimal balance, preventing the search tree from wasting time on dead-end branches in the wider multi-agent state space, while fully exploiting the high-reward paths found by the rollout policy).
*   **max_depth (Rollout Horizon):** 30 - It should allow for most cases to get to a goal
*   **gamma (Discount Factor):** 0.99 - Like in previous tasks
*   **Rollout Policy:** Dynamic "Nearest Box" heuristic with a 20% probability of taking a random action in order to keep exploration alive.

## 3. Results Table
The following metrics were aggregated over 30 independent runs (N=30) for each configuration.

| Scenario | Time Budget | Mean Steps | Standard Deviation | Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Single Agent** | 1.0s | 12.53 | 3.08 | 100% (30/30) |
| **Single Agent** | 20.0s | 11.80 | 2.57 | 100% (30/30) |
| **Multi-Agent** | 1.0s | 10.53 | 2.16 | 100% (30/30) |
| **Multi-Agent** | 20.0s | 10.40 | 2.67 | 100% (30/30) |

The first run of each was rendered and saved in the final_run.txt file.

## 4. Discussion of Results

**Impact of Time Budget (1s vs. 20s):**
Increasing the time budget from 1.0 second to 20.0 seconds per decision did yield measurable improvements, reducing the average step count from 12.53 to 11.80 for the single agent, and from 10.53 to 10.40 for the multi-agent configuration. However, given the 20x increase in computational time, this improvement is relatively marginal. This demonstrates that the rollout heuristic is highly efficient. The UCT search does not require deep, exhaustive simulations to converge on a high-quality action, as the heuristic effectively guides the value estimation almost immediately within the shorter 1.0s budget. Ultimately, allocating more time provides diminishing returns, though the stable standard deviations confirm that the longer budget safely avoided inducing any search pathology.

**Multi-Agent vs. Single-Agent Performance:**
Generally, multi-agent environments are significantly more complex to solve than single-agent ones. Multiple agents typically face severe physical bottlenecks, struggle with deadlocks, and must navigate an exponentially larger joint-action state space. Furthermore, comparing our two specific scenarios directly is not entirely fair: the multi-agent setup introduces a heavy box (`C`) that explicitly requires both agents to coordinate and execute a joint push, a complex mechanic that is entirely absent from the single-agent task.

However, because the two starting maps are structurally almost identical, we can still draw meaningful comparisons between their performances:

**Single-Agent Map:**

```text
WWWWWWWW
WGB    W
W      W
W B  A W
W      W
W G    W
WWWWWWWW

```

**Multi-Agent Map:**

```text
WWWWWWWW
WGC    W
W    A W
W B  A W
W      W
W G    W
WWWWWWWW

```

Both environments share the exact same 8x7 grid layout, outer wall placements (`W`), and general goal locations (`G`). Despite the added difficulty of the heavy box and the high potential for coordination bottlenecks, the multi-agent configuration actually outperformed the single-agent setup, achieving the lowest overall mean of 10.40 steps versus 11.80 steps on the 20.0s budget.

We can deduce that this improvement occurs because the second agent enables parallel task execution. While the time taken to jointly push the heavy box (`C`) at the top remains somewhat similar to the single-agent pushing its top box, the agents can divide the remaining work, reducing the variance and time required to navigate to and push the lower, non-heavy boxes (`B`).
## 5. Advanced Logical Optimizations
To maximize the efficiency of the POMCP algorithm within the strict time budgets, several domain-specific logical enhancements were implemented:

### Static Deadlock Analysis (Reverse BFS)
In Grid/Sokoban environments, a major inefficiency in Monte Carlo simulations is the tendency to push boxes into unrecoverable positions (e.g., corners or flat walls). To solve this, the environment pre-computes a set of `dead_cells` during initialization. By performing a reverse Breadth-First Search (BFS) starting from the goals, the map identifies all floor cells from which a box can never legally reach a goal. This static map knowledge prevents the tree search from wasting budget exploring permanently failed game states.

### Dynamic "Absolute Nearest Box" Rollout Heuristic
While a purely random rollout policy worked great for a single agent, it proved insufficient for the multi-agent scenario. Multiple agents using a random policy could not successfully coordinate to push the large box together, frequently resulting in deadlocks and truncated episodes. Instead of relying on this purely random policy, which degrades rapidly in wide multi-agent state spaces, a custom goal-directed heuristic was implemented:
* **Target Claiming & Penalty:** To prevent agents from deadlocking in hallways or fighting over the same box, the heuristic implements a dynamic claiming system. When an agent targets a box, it applies a massive distance penalty (e.g., +1000) to that box for the other agent, forcing them to divide and conquer.
* **Dead-Cell Avoidance:** When the heuristic calculates the optimal push direction for a box, it actively verifies that the push destination is not in the pre-computed `dead_cells` set. This ensures the rollout policy only simulates viable, productive paths, leading to highly accurate value estimates even on the 1.0s time budget.

