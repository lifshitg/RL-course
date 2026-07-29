"""
Environment logic, generative model, and observation functions.
"""

from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from config import POMCPConfig

# ===========================================================================
# Type Aliases & Constants
# ===========================================================================
Pos = Tuple[int, int]
Window = Tuple[Tuple[int, int, int], ...]
State = Tuple[Tuple[Pos, ...], FrozenSet[Pos], Optional[Pos]]

# Observation cell types
CELL_FREE = 0
CELL_WALL = 1
CELL_BOX = 2
OOB = CELL_WALL  # Outside the map border reads as wall

# Movement dictionaries
DIRS: Dict[int, Pos] = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}

# Perpendicular deviations for stochastic moves (0.1 / 0.1)
PERP: Dict[int, Tuple[int, int]] = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}


# ===========================================================================
# Map Parsing & Static Data
# ===========================================================================
class MapData:
    """Static map info: walls, goals, initial boxes/agents, and free cells."""

    def __init__(self, ascii_map: List[str]):
        self.height = len(ascii_map)
        self.width = len(ascii_map[0])
        self.walls = np.zeros((self.height, self.width), dtype=bool)

        self.agent_starts: List[Pos] = []
        self.heavy_start: Optional[Pos] = None

        small: List[Pos] = []
        goals: List[Pos] = []

        # Parse ASCII map
        for y, row in enumerate(ascii_map):
            for x, ch in enumerate(row):
                if ch == "W":
                    self.walls[y, x] = True
                elif ch == "A":
                    self.agent_starts.append((x, y))
                elif ch == "B":
                    small.append((x, y))
                elif ch == "C":
                    self.heavy_start = (x, y)
                elif ch == "G":
                    goals.append((x, y))

        self.small_start: FrozenSet[Pos] = frozenset(small)
        self.goals: FrozenSet[Pos] = frozenset(goals)
        self.n_agents = len(self.agent_starts)

        # Determine valid floor cells (ignoring dynamic box positions)
        self.floor_cells: List[Pos] = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if not self.walls[y, x]
        ]

        # Pre-compute statically dead cells for the planning heuristic
        self.dead_cells: FrozenSet[Pos] = self._compute_dead_cells()

    def _compute_dead_cells(self) -> FrozenSet[Pos]:
        """
        Static Sokoban-style deadlock analysis.
        A cell is LIVE if a box there can still reach a goal via legal pushes.
        """

        def ok(p: Pos) -> bool:
            x, y = p
            return (0 <= x < self.width and 0 <= y < self.height and not self.walls[y, x])

        live = set(self.goals)
        frontier = list(self.goals)

        # Reverse BFS from goals over the push graph
        while frontier:
            t = frontier.pop()
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                c = (t[0] - dx, t[1] - dy)  # Box position before push
                agent = (c[0] - dx, c[1] - dy)  # Agent push position
                if ok(c) and ok(agent) and c not in live:
                    live.add(c)
                    frontier.append(c)

        return frozenset(c for c in self.floor_cells if c not in live)

    def valid_agent_cells(self, small: FrozenSet[Pos], heavy: Optional[Pos]) -> List[Pos]:
        """Returns floor cells currently unoccupied by any boxes."""
        boxes = set(small)
        if heavy is not None:
            boxes.add(heavy)
        return [c for c in self.floor_cells if c not in boxes]


# ===========================================================================
# Observation Function
# ===========================================================================
def get_observation(agent_pos: Pos, map_data: MapData,
                    small: FrozenSet[Pos], heavy: Optional[Pos]) -> Window:
    """
    Returns an egocentric 3x3 window centered on the agent's position.
    """
    x, y = agent_pos
    h, w = map_data.height, map_data.width
    heavy_set = {heavy} if heavy is not None else set()

    window = []
    for dy in (-1, 0, 1):
        row = []
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                row.append(CELL_FREE)
                continue

            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                row.append(OOB)
            elif map_data.walls[ny, nx]:
                row.append(CELL_WALL)
            elif (nx, ny) in small or (nx, ny) in heavy_set:
                row.append(CELL_BOX)
            else:
                row.append(CELL_FREE)
        window.append(tuple(row))

    return tuple(window)


# ===========================================================================
# Generative Model (Stochastic Dynamics)
# ===========================================================================
class GenerativeModel:
    """
    Stochastic dynamics used by both POMCP planning and particle filter updates.

    State Representation
    --------------------
    The state is a deterministic, hashable tuple: `(agents, small, heavy)`
      * agents : Tuple[Pos, ...] - A tuple of (x, y) coordinates for each agent.
                 Length is 1 (single-agent) or 2 (multi-agent).
      * small  : FrozenSet[Pos] - An immutable set of (x, y) coordinates for all small boxes.
      * heavy  : Optional[Pos] - The (x, y) coordinate of the heavy box, or None if absent.

    Action Representation
    ---------------------
    The environment expects a single integer representing the joint action of all agents.
    Base directions are: 0=North, 1=South, 2=East, 3=West.
      * Single-Agent : An integer [0, 3] representing the agent's move.
      * Multi-Agent  : An integer [0, 15] using base-4 encoding.
                       Decoded as: Agent 0 = action // 4, Agent 1 = action % 4.
    """

    def __init__(self, map_data: MapData, config: POMCPConfig):
        self.map = map_data
        self.config = config
        self.n_agents = map_data.n_agents
        self.n_actions = 4 ** self.n_agents
        self._obs_cache: Dict[Tuple, Tuple[Window, ...]] = {}

    # --- Core Helpers ---
    def decode_action(self, action: int) -> Tuple[int, ...]:
        """Converts a joint action index to per-agent base-4 direction codes."""
        if self.n_agents == 1:
            return (action,)
        return (action // 4, action % 4)

    def _cell_blocked(self, pos: Pos, small: FrozenSet[Pos], heavy: Optional[Pos]) -> bool:
        """Checks if a cell is blocked by a wall, out of bounds, or a box."""
        x, y = pos
        if not (0 <= x < self.map.width and 0 <= y < self.map.height):
            return True
        if self.map.walls[y, x]:
            return True
        if pos in small or (heavy is not None and pos == heavy):
            return True
        return False

    def is_terminal(self, state: State) -> bool:
        """Returns True if all goals are covered by boxes."""
        _, small, heavy = state
        boxes = set(small)
        if heavy is not None:
            boxes.add(heavy)
        return self.map.goals <= boxes

    def is_deadlocked(self, state: State) -> bool:
        """Returns True if any box is on a statically dead cell."""
        _, small, heavy = state
        dead = self.map.dead_cells
        if heavy is not None and heavy in dead:
            return True
        return any(b in dead for b in small)

    def observe(self, state: State) -> Tuple[Window, ...]:
        """Returns deterministic, cached joint observations for all agents."""
        agents, small, heavy = state
        key = (agents, small, heavy)

        cached = self._obs_cache.get(key)
        if cached is None:
            cached = tuple(get_observation(p, self.map, small, heavy) for p in agents)
            self._obs_cache[key] = cached

        return cached

    # --- Action Resolution ---
    def _apply_one(self, pos: Pos, d: int, small: FrozenSet[Pos],
                   heavy: Optional[Pos], rng: random.Random) -> Tuple[Pos, FrozenSet[Pos]]:
        """Resolves a single agent's move/push for small boxes only."""
        dx, dy = DIRS[d]
        target = (pos[0] + dx, pos[1] + dy)
        tx, ty = target

        # Handle out of bounds or walls
        in_bounds = 0 <= tx < self.map.width and 0 <= ty < self.map.height
        if not in_bounds or self.map.walls[ty, tx]:
            return pos, small

        # Handle pushing a small box
        if target in small:
            beyond = (target[0] + dx, target[1] + dy)
            if self._cell_blocked(beyond, small, heavy):
                return pos, small
            if rng.random() < 0.8:
                new_small = (small - {target}) | {beyond}
                return target, frozenset(new_small)
            return pos, small

        # Ignore heavy box interactions for single agents
        if heavy is not None and target == heavy:
            return pos, small

        # Handle stochastic movement
        r = rng.random()
        if r < 0.8:
            actual = d
        elif r < 0.9:
            actual = PERP[d][0]
        else:
            actual = PERP[d][1]

        ax, ay = DIRS[actual]
        dest = (pos[0] + ax, pos[1] + ay)

        if self._cell_blocked(dest, small, heavy):
            return pos, small
        return dest, small

    # --- Generative Step ---
    def step(self, state: State, action: int, rng: random.Random) -> Tuple[State, Tuple[Window, ...], float, bool]:
        """Samples one outcome step based on joint actions."""
        agents, small, heavy = state
        dirs = self.decode_action(action)

        if self.n_agents == 2:
            p0, p1 = agents
            d0, d1 = dirs

            # Joint heavy box push logic
            if heavy is not None and p0 == p1 and d0 == d1:
                dx, dy = DIRS[d0]
                target = (p0[0] + dx, p0[1] + dy)
                if target == heavy:
                    beyond = (target[0] + dx, target[1] + dy)
                    if not self._cell_blocked(beyond, small, None):
                        if rng.random() < 0.8:
                            heavy = beyond
                            p0 = p1 = target

                    next_state = ((p0, p1), small, heavy)
                    obs = self.observe(next_state)
                    done = self.is_terminal(next_state)
                    return next_state, obs, self.config.step_reward, done

            # Sequential resolution if not a joint push
            p0, small = self._apply_one(p0, d0, small, heavy, rng)
            p1, small = self._apply_one(p1, d1, small, heavy, rng)
            next_state = ((p0, p1), small, heavy)

        else:
            # Single agent logic
            (p0,) = agents
            p0, small = self._apply_one(p0, dirs[0], small, heavy, rng)
            next_state = ((p0,), small, heavy)

        # Return standardized outputs
        obs = self.observe(next_state)
        done = self.is_terminal(next_state)
        reward = self.config.step_reward + (self.config.goal_reward if done else 0.0)

        return next_state, obs, reward, done


# ===========================================================================
# True Environment Simulator
# ===========================================================================
class TrueEnv:
    """
    Manages the TRUE (hidden) state, exposing only observations and box info.
    """

    def __init__(self, model: GenerativeModel, config: POMCPConfig, rng: random.Random):
        self.model = model
        self.config = config
        self.rng = rng
        self._state: Optional[State] = None
        self.steps = 0

    def reset(self):
        """Resets environment utilizing static map start positions."""
        m = self.model.map
        small = m.small_start
        heavy = m.heavy_start
        agents = tuple(m.agent_starts)

        self._state = (agents, small, heavy)
        self.steps = 0

        obs = self.model.observe(self._state)
        info = {"small": small, "heavy": heavy}
        return obs, info

    def step(self, action: int):
        """Advances the true state by one step."""
        next_state, obs, reward, done = self.model.step(self._state, action, self.rng)
        self._state = next_state
        self.steps += 1

        truncated = self.steps >= self.config.max_env_steps and not done
        info = {"small": next_state[1], "heavy": next_state[2]}

        return obs, reward, done, truncated, info

    @property
    def true_state(self) -> State:
        """Exposes true state for diagnostics and testing only."""
        return self._state
