"""
Configuration and static maps for the POMDP environment.
"""

from dataclasses import dataclass
from typing import List

# ===========================================================================
# Static Maps
# ===========================================================================
SINGLE_AGENT_MAP = [
    "WWWWWWWW",
    "WGB    W",
    "W      W",
    "W B  A W",
    "W      W",
    "W G    W",
    "WWWWWWWW",
]

MULTI_AGENT_MAP = [
    "WWWWWWWW",
    "WGC    W",
    "W    A W",
    "W B  A W",
    "W      W",
    "W G    W",
    "WWWWWWWW",
]

MAPS = {
    "single_agent": SINGLE_AGENT_MAP,
    "multi_agent": MULTI_AGENT_MAP,
}


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class POMCPConfig:
    """Hyperparameters for the POMCP experiment."""

    # --- Experiment & Environment Settings ---
    scenario: str = "single_agent"  # Scenario: "single_agent" or "multi_agent"
    seed: int = 42  # Random seed for reproducibility
    n_runs: int = 30  # Number of evaluation runs
    max_env_steps: int = 100  # Maximum steps per episode

    # --- Planning Time Budgets ---
    time_budget_short: float = 1.0  # Short planning budget in seconds
    time_budget_long: float = 20.0  # Long planning budget in seconds

    # --- Particle Filter Settings ---
    n_particles: int = 500  # Number of particles for the belief state
    max_rejection_tries: int = 20_000  # Max attempts for rejection sampling

    # --- POMCP & Rollout Parameters ---
    ucb_c: float = 2.0  # UCB1 exploration constant
    rollout_depth: int = 30  # Maximum horizon for rollout simulation
    rollout_random_prob: float = 0.2  # For heuristic rollout exploration
    gamma: float = 0.99  # Discount factor for future rewards

    # --- Reward Structure ---
    step_reward: float = -1.0  # Penalty per step to encourage speed
    goal_reward: float = 0.0  # Reward upon reaching the terminal state

    @property
    def ascii_map(self) -> List[str]:
        """Returns the corresponding ASCII map based on the active scenario."""
        return MAPS[self.scenario]
