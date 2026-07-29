"""
Runner file for POMDP / POMCP experiments.
Executes the main loops.
"""
import random
import time
from typing import Tuple

import numpy as np

# Import configuration
from config import POMCPConfig

# Import environment components
from environment import MapData, GenerativeModel, TrueEnv, State

# Import planning and belief tracking components
from pomdp import ParticleFilter, POMCP

import sys


class Tee:
    """Helper class to duplicate stdout to both the console and a UTF-8 file (overwriting it from scratch)."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        # Open with encoding='utf-8' to prevent any character corruption
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self.log:
            self.log.close()


def render_frame(config: POMCPConfig, state, step: int, action=None):
    """Prints a visual representation of the current true environment state."""
    # 1. Create a clean base map by stripping out the initial dynamic entities
    base_map = []
    for row in config.ascii_map:
        clean_row = row.replace('A', ' ').replace('B', ' ').replace('C', ' ')
        base_map.append(list(clean_row))

    # 2. Overlay the current dynamic entities from the true state
    for (x, y) in state[1]:  # small boxes
        base_map[y][x] = 'B'

    heavy = state[2]
    if heavy is not None:
        base_map[heavy[1]][heavy[0]] = 'C'

    # Agent Overlap Detection
    agent_positions = {}
    for (x, y) in state[0]:
        agent_positions[(x, y)] = agent_positions.get((x, y), 0) + 1

    for (x, y), count in agent_positions.items():
        # If multiple agents are on the same cell, print the number instead of 'A'
        base_map[y][x] = str(count) if count > 1 else 'A'

    # 3. Print the frame with perfectly aligned borders
    action_str = f"{action}" if action is not None else 'START'
    print(f"\n  [ Step {step} | Action: {action_str} ]")

    map_width = len(base_map[0])
    print("  ╭" + "─" * (map_width + 2) + "╮")

    for row in base_map:
        # Replace 'W' with a block character like '█' for a cleaner look
        row_str = "".join(row).replace('W', '█')
        print(f"  │ {row_str} │")

    print("  ╰" + "─" * (map_width + 2) + "╯")

    # Add a small delay to make it watchable as a "gameplay" feed
    time.sleep(0.3)


def run_single_episode(config: POMCPConfig, time_budget: float, seed: int,
                       verbose: bool = False) -> Tuple[int, bool]:
    """One full POMDP episode; returns (steps_taken, is_success)."""
    rng = random.Random(seed)
    map_data = MapData(config.ascii_map)
    model = GenerativeModel(map_data, config)
    env = TrueEnv(model, config, rng)
    pomcp = POMCP(model, config, rng)
    pf = ParticleFilter(model, config, rng)

    obs, info = env.reset()
    small, heavy = info["small"], info["heavy"]
    pf.init_uniform(small, heavy)

    # The very first observation arrives before any action; filter the
    # uniform prior down to the positions consistent with it.
    per_agent = [pf._consistent_cells(w, small, heavy) for w in obs]
    per_agent = [c if c else map_data.valid_agent_cells(small, heavy)
                 for c in per_agent]
    pf.particles = [
        tuple(rng.choice(c) for c in per_agent)
        for _ in range(config.n_particles)
    ]

    steps = 0
    terminated = False
    truncated = False
    done = False

    if verbose:
        render_frame(config, env.true_state, steps, action=None)

    while not done and steps < config.max_env_steps:
        action = pomcp.search(pf.particles, small, heavy, time_budget)
        obs, reward, terminated, truncated, info = env.step(action)
        pf.update(action, obs, small, heavy, info["small"], info["heavy"])
        small, heavy = info["small"], info["heavy"]
        steps += 1
        done = terminated or truncated

        if verbose:
            render_frame(config, env.true_state, steps, action=action)
            uniq = len(set(pf.particles))
            print(
                f"      [Step {steps:>3}] Action: {action:<2} | Sims: {pomcp.last_n_sims:<5} | Belief Support: {uniq}")

    return steps, terminated


def run_full_experiment(config: POMCPConfig, scenarios=None, budgets=None,
                        verbose_first: bool = False):
    """
    For each scenario x time budget: run `config.n_runs` episodes and report
    mean +/- std of steps-to-solve, plus success rate. Prints a structured summary table.
    """
    scenarios = scenarios or ["single_agent", "multi_agent"]
    budgets = budgets or [config.time_budget_short, config.time_budget_long]
    results = {}

    for scenario in scenarios:
        cfg = POMCPConfig(**{**config.__dict__, "scenario": scenario})
        for budget in budgets:
            key = (scenario, budget)

            # Formatted Header
            print(f"\n╭" + "─" * 70 + "╮")
            print(f"│ SCENARIO: {scenario:<15} | BUDGET: {budget:>5}s | RUNS: {cfg.n_runs:<5}             │")
            print(f"╰" + "─" * 70 + "╯")

            steps_list = []
            success_count = 0

            for run in range(cfg.n_runs):
                t0 = time.perf_counter()
                steps, is_success = run_single_episode(
                    cfg, budget, seed=cfg.seed + run,
                    verbose=(verbose_first and run == 0),
                )
                dt = time.perf_counter() - t0
                steps_list.append(steps)

                # Check outcome status
                if is_success:
                    success_count += 1
                    status_str = "✓ Success"
                else:
                    status_str = "✗ Timeout"

                # Run Tracking
                print(
                    f"  ├─ Run {run + 1:>2}/{cfg.n_runs} ➔  Steps: {steps:<4} | Time: {dt:>5.1f}s | Status: {status_str}")

            arr = np.array(steps_list, dtype=float)
            success_rate = (success_count / cfg.n_runs) * 100
            results[key] = (arr.mean(), arr.std(), success_count, cfg.n_runs, success_rate)

            # Batch Summary
            print(
                f"  ╰─ RESULTS ➔  Mean: {arr.mean():>6.2f} steps | Std: {arr.std():>5.2f} | Success: {success_rate:.0f}%\n")

    # Final Formatted Summary Table
    print("═" * 80)
    print(f"│ {'POMCP EXPERIMENT SUMMARY':^76} │")
    print("═" * 80)
    print(f"│ {'Scenario':<12} │ {'Budget':>8} │ {'Mean Steps':>12} │ {'Std Dev':>9} │ {'Success Rate':>18} │")
    print("├" + "─" * 14 + "┼" + "─" * 10 + "┼" + "─" * 14 + "┼" + "─" * 11 + "┼" + "─" * 20 + "┤")

    for (scenario, budget), (mean, std, succ, total, rate) in results.items():
        rate_str = f"{rate:>3.0f}% ({succ}/{total})"
        print(f"│ {scenario:<12} │ {budget:>7.1f}s │ {mean:>12.2f} │ {std:>9.2f} │ {rate_str:>18} │")

    print("═" * 80)
    return results

def c_grid_search():
    print("=== STARTING COMPREHENSIVE C-PARAMETER GRID SEARCH ===")

    c_values_to_test = [0.5, 1.0, 2.0, 5.0, 10.0]
    scenarios_to_test = ["single_agent", "multi_agent"]

    for scenario in scenarios_to_test:
        print(f"\n\n########################################")
        print(f" SCENARIO: {scenario.upper()}")
        print(f"########################################")

        for c_val in c_values_to_test:
            print(f"\n{'-' * 40}")
            print(f" Testing Scenario: {scenario} | UCB1 Constant: c = {c_val}")
            print(f"{'-' * 40}\n")

            # 1. Initialize config without kwargs
            config = POMCPConfig()

            # 2. Assign parameters as direct attributes to avoid TypeError
            config.ucb_c = c_val
            config.n_runs = 5

            run_full_experiment(
                config,
                scenarios=[scenario],
                verbose_first=False
            )

    print("\n\n" + "=" * 50)
    print(" GRID SEARCH EXPERIMENT SUMMARY")
    print("=" * 50)
    print("All scenarios and c-values have completed execution.")
    print("Tested c values:", c_values_to_test)
    print("Tested scenarios:", scenarios_to_test)
    print("Review the block metrics above or check 'exercises/ex4/output.txt'")
    print("to compare success rates, steps-to-completion, and reward averages.")
    print("=" * 50)

# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    # Redirect stdout to our Tee class pointing to output.txt (overwrites from scratch)
    tee_output = Tee("exercises/ex4/output.txt")
    sys.stdout = tee_output

    try:
        # ["single_agent", "multi_agent"]
        run_full_experiment(POMCPConfig(), scenarios=["single_agent", "multi_agent"], verbose_first=True)

        c_grid_search()
    finally:
        # Ensure the file safely closes even if the script finishes or errors out
        sys.stdout = tee_output.terminal
        tee_output.close()