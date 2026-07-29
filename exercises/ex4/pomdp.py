"""
Particle Filter and POMCP planner logic.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, FrozenSet, List, Optional, Tuple

from config import POMCPConfig
from environment import GenerativeModel, Pos, State, Window, get_observation, DIRS


# ===========================================================================
# Particle Filter (Belief over agent positions)
# ===========================================================================
class ParticleFilter:
    """
    Belief state = N particles; each particle is a tuple of agent positions
    (length 1 or 2). Box positions are NOT part of the particle -- they are
    fully known and supplied externally at each update.
    """

    def __init__(self, model: GenerativeModel, config: POMCPConfig, rng: random.Random):
        self.model = model
        self.config = config
        self.rng = rng
        self.particles: List[Tuple[Pos, ...]] = []

    def init_uniform(self, small: FrozenSet[Pos], heavy: Optional[Pos]):
        """Scatter particles uniformly over all valid (box-free) cells."""
        cells = self.model.map.valid_agent_cells(small, heavy)
        n_agents = self.model.n_agents
        self.particles = [
            tuple(self.rng.choice(cells) for _ in range(n_agents))
            for _ in range(self.config.n_particles)
        ]

    def _consistent_cells(self, window: Window, small: FrozenSet[Pos], heavy: Optional[Pos]) -> List[Pos]:
        """All cells whose egocentric window matches `window` exactly."""
        return [
            c for c in self.model.map.valid_agent_cells(small, heavy)
            if get_observation(c, self.model.map, small, heavy) == window
        ]

    def update(self, action: int, real_obs: Tuple[Window, ...],
               small_before: FrozenSet[Pos], heavy_before: Optional[Pos],
               small_after: FrozenSet[Pos], heavy_after: Optional[Pos]):
        """
        Unweighted rejection-sampling update (Silver & Veness 2010), using
        the SAME generative model POMCP plans with.

        If the filter cannot be refilled within max_rejection_tries, it falls
        back to particle reinvigoration using full map knowledge to directly
        compute the set of positions consistent with the real observation.
        """
        n = self.config.n_particles
        new: List[Tuple[Pos, ...]] = []
        tries = 0
        max_tries = self.config.max_rejection_tries

        # 1. Standard Rejection Sampling
        while len(new) < n and tries < max_tries:
            tries += 1
            hypo = self.rng.choice(self.particles)
            state: State = (hypo, small_before, heavy_before)

            nxt, sim_obs, _, _ = self.model.step(state, action, self.rng)

            if (nxt[1] == small_after and nxt[2] == heavy_after and sim_obs == real_obs):
                new.append(nxt[0])

        # 2. Reinvigoration Fallback
        if len(new) < n:
            per_agent = [
                self._consistent_cells(w, small_after, heavy_after)
                for w in real_obs
            ]

            # Safety net (cannot normally happen with a correct model)
            per_agent = [
                cells if cells else self.model.map.valid_agent_cells(small_after, heavy_after)
                for cells in per_agent
            ]

            while len(new) < n:
                new.append(tuple(self.rng.choice(c) for c in per_agent))

        self.particles = new


# ===========================================================================
# POMCP Planner Data Structures
# ===========================================================================
class POMCPNode:
    """Statistics for one history node: visit count + per-action stats."""

    def __init__(self, n_actions: int):
        self.n_visits = 0
        self.n_a = [0] * n_actions
        self.v_a = [0.0] * n_actions


# ===========================================================================
# POMCP Planner Core
# ===========================================================================
class POMCP:
    """
    Monte-Carlo tree search over action-observation histories, run directly
    on the particle belief. The per-decision time budget is ENFORCED inside
    the planning loop.
    """

    def __init__(self, model: GenerativeModel, config: POMCPConfig, rng: random.Random):
        self.model = model
        self.config = config
        self.rng = rng
        self.tree: Dict[Tuple, POMCPNode] = {}
        self.last_n_sims = 0
        self.max_depth = config.rollout_depth

    # --- Public API ---
    def search(self, particles: List[Tuple[Pos, ...]],
               small: FrozenSet[Pos], heavy: Optional[Pos],
               time_budget: float) -> int:
        """Run simulations until `time_budget` elapses; return best action."""
        self.tree = {}  # Fresh tree each decision
        root: Tuple = ()
        self.tree[root] = POMCPNode(self.model.n_actions)

        t0 = time.perf_counter()
        n_sims = 0

        # Planning loop bound by real time
        while time.perf_counter() - t0 < time_budget:
            hypo = self.rng.choice(particles)
            state: State = (hypo, small, heavy)

            if self.model.is_terminal(state):
                n_sims += 1
                continue

            self._simulate(state, root, 0)
            n_sims += 1

        self.last_n_sims = n_sims

        # Action selection from the root node
        node = self.tree[root]
        best_a, best_v = 0, -float("inf")

        for a in range(self.model.n_actions):
            if node.n_a[a] > 0 and node.v_a[a] > best_v:
                best_a, best_v = a, node.v_a[a]

        # Failsafe if no simulations completed (tiny budget)
        if best_v == -float("inf"):
            best_a = self.rng.randrange(self.model.n_actions)

        return best_a

    # --- Internals ---
    def _ucb_select(self, node: POMCPNode) -> int:
        """Selects an action using the UCB1 algorithm."""
        untried = [a for a in range(self.model.n_actions) if node.n_a[a] == 0]
        if untried:
            return self.rng.choice(untried)

        log_n = math.log(node.n_visits)
        c = self.config.ucb_c
        best_a, best_score = 0, -float("inf")

        for a in range(self.model.n_actions):
            score = node.v_a[a] + c * math.sqrt(log_n / node.n_a[a])
            if score > best_score:
                best_a, best_score = a, score

        return best_a

    def _simulate(self, state: State, h: Tuple, depth: int) -> float:
        """Recursively simulates state transitions to build the search tree."""
        if depth >= self.max_depth or self.model.is_terminal(state):
            return 0.0

        node = self.tree.get(h)
        if node is None:
            # New history: add the node, estimate its value with a rollout.
            self.tree[h] = POMCPNode(self.model.n_actions)
            return self._rollout(state, depth)

        a = self._ucb_select(node)
        next_state, obs, reward, done = self.model.step(state, a, self.rng)

        if done:
            ret = reward
        else:
            child_h = h + ((a, obs),)
            ret = reward + self.config.gamma * self._simulate(next_state, child_h, depth + 1)

        node.n_visits += 1
        node.n_a[a] += 1
        node.v_a[a] += (ret - node.v_a[a]) / node.n_a[a]
        return ret

    def _rollout(self, state: State, depth: int) -> float:
        """
        Performs a heuristic rollout simulation from a concrete state
        up to the maximum rollout depth to estimate the value of unvisited nodes.
        """
        total, discount = 0.0, 1.0
        d, s = depth, state

        while d < self.max_depth and not self.model.is_terminal(s):
            a = self._rollout_action(s)
            s2, _, r, done = self.model.step(s, a, self.rng)

            total += discount * r
            discount *= self.config.gamma
            s = s2
            d += 1

            if done:
                break

        return total

    # --- Heuristic Rollout Policy ---
    def _rollout_action(self, state) -> int:
        """
        Determines the next action during a heuristic rollout.
        Dynamically targets the nearest box, splitting multi-agent targets
        organically to prevent hallway deadlocks.
        """
        cfg = self.config

        # 1. Exploration check: occasional random action
        if self.rng.random() < cfg.rollout_random_prob:
            return self.rng.randrange(self.model.n_actions)

        agents, small_boxes, heavy_box = state
        map_data = self.model.map

        # 2. Identify active goals and combine ALL unsolved boxes
        all_unsolved_boxes = [b for b in small_boxes if b not in map_data.goals]
        if heavy_box is not None and heavy_box not in map_data.goals:
            all_unsolved_boxes.append(heavy_box)

        free_goals = [g for g in map_data.goals if g not in small_boxes and g != heavy_box]

        agent_directions = []
        claimed_boxes = set()

        # 3. Determine each agent's target dynamically
        for agent_pos in agents:
            push_plan = None

            if all_unsolved_boxes and free_goals:
                # Score boxes by distance. Add a massive penalty if another agent
                # already claimed a small box, forcing this agent to find a new target.
                def box_score(box):
                    dist = abs(box[0] - agent_pos[0]) + abs(box[1] - agent_pos[1])
                    if box in claimed_boxes and box != heavy_box:
                        dist += 1000
                    return dist

                # Find nearest box and claim it
                nearest_box = min(all_unsolved_boxes, key=box_score)
                claimed_boxes.add(nearest_box)

                push_plan = self._find_push_plan(nearest_box, free_goals, small_boxes, heavy_box)

            # Fallback: Move randomly if no paths are viable
            if push_plan is None:
                agent_directions.append(self.rng.randrange(4))
                continue

            # Execute push if in position, otherwise navigate
            target_push_dir, target_push_pos = push_plan
            if agent_pos == target_push_pos:
                agent_directions.append(target_push_dir)
            else:
                agent_directions.append(self._get_greedy_step(agent_pos, target_push_pos, small_boxes, heavy_box))

        # 4. Encode and return the joint action index
        if self.model.n_agents == 1:
            return agent_directions[0]
        return agent_directions[0] * 4 + agent_directions[1]

    # def _rollout_action(self, state: State) -> int:
    #     """
    #     Determines the next action during a heuristic rollout using a goal-directed policy.
    #
    #     Priority & Behavior Hierarchy:
    #     1. Exploration Check: Occasional random action based on rollout_random_prob.
    #     2. Heavy Box Priority: Pre-plans a pushing route for unsolved heavy boxes.
    #     3. Small Box Fallback: Targets nearest unsolved small boxes if heavy is clear.
    #     4. Navigation: Moves greedily towards push positions, or pushes if in position.
    #     """
    #     cfg = self.config
    #
    #     # 1. Exploration check: occasional random action
    #     if self.rng.random() < cfg.rollout_random_prob:
    #         return self.rng.randrange(self.model.n_actions)
    #
    #     agents, small_boxes, heavy_box = state
    #     map_data = self.model.map
    #
    #     # Identify active goals and unsolved boxes
    #     all_boxes = set(small_boxes) | ({heavy_box} if heavy_box is not None else set())
    #     free_goals = [g for g in map_data.goals if g not in all_boxes]
    #
    #     unsolved_small = [b for b in small_boxes if b not in map_data.goals]
    #     is_heavy_unsolved = heavy_box is not None and heavy_box not in map_data.goals
    #
    #     # 2. Pre-plan the heavy box push first
    #     heavy_push_plan = None
    #     if is_heavy_unsolved and free_goals:
    #         heavy_push_plan = self._find_push_plan(heavy_box, free_goals, small_boxes, heavy_box)
    #
    #     agent_directions: List[int] = []
    #
    #     # 3. Determine each agent's target
    #     for agent_pos in agents:
    #         push_plan = None
    #
    #         # Priority 1: Heavy box
    #         if heavy_push_plan is not None:
    #             push_plan = heavy_push_plan
    #
    #         # Priority 2: Nearest unsolved small box
    #         if push_plan is None and unsolved_small and free_goals:
    #             nearest_small_box = min(
    #                 unsolved_small,
    #                 key=lambda box: abs(box[0] - agent_pos[0]) + abs(box[1] - agent_pos[1])
    #             )
    #             push_plan = self._find_push_plan(nearest_small_box, free_goals, small_boxes, heavy_box)
    #
    #         # Fallback: Move randomly if no plans found
    #         if push_plan is None:
    #             agent_directions.append(self.rng.randrange(4))
    #             continue
    #
    #         # Execute push if in position, otherwise navigate
    #         target_push_dir, target_push_pos = push_plan
    #         if agent_pos == target_push_pos:
    #             agent_directions.append(target_push_dir)
    #         else:
    #             agent_directions.append(self._get_greedy_step(agent_pos, target_push_pos, small_boxes, heavy_box))
    #
    #     # 4. Encode and return the joint action index
    #     if self.model.n_agents == 1:
    #         return agent_directions[0]
    #     return agent_directions[0] * 4 + agent_directions[1]

    def _find_push_plan(self, box_pos: Pos, free_goals: List[Pos],
                        small_boxes: FrozenSet[Pos], heavy_box: Optional[Pos]) -> Optional[Tuple[int, Pos]]:
        """Finds the optimal (direction, push_position) to move a box toward its nearest free goal."""
        nearest_goal = min(
            free_goals,
            key=lambda goal: abs(goal[0] - box_pos[0]) + abs(goal[1] - box_pos[1])
        )

        candidate_dirs = []
        if nearest_goal[0] > box_pos[0]: candidate_dirs.append(2)  # Push East
        if nearest_goal[0] < box_pos[0]: candidate_dirs.append(3)  # Push West
        if nearest_goal[1] > box_pos[1]: candidate_dirs.append(1)  # Push South
        if nearest_goal[1] < box_pos[1]: candidate_dirs.append(0)  # Push North

        self.rng.shuffle(candidate_dirs)

        for direction in candidate_dirs:
            dx, dy = DIRS[direction]
            destination = (box_pos[0] + dx, box_pos[1] + dy)

            # Validate destination
            if self.model._cell_blocked(destination, small_boxes, heavy_box):
                continue
            if destination in self.model.map.dead_cells:
                continue

            push_position = (box_pos[0] - dx, box_pos[1] - dy)
            px, py = push_position

            # Validate push position
            if not (0 <= px < self.model.map.width and 0 <= py < self.model.map.height):
                continue
            if self.model.map.walls[py, px] or push_position in small_boxes or (
                    heavy_box is not None and push_position == heavy_box):
                continue

            return direction, push_position

        return None

    def _get_greedy_step(self, agent_pos: Pos, target_pos: Pos,
                         small_boxes: FrozenSet[Pos], heavy_box: Optional[Pos]) -> int:
        """Calculates a single greedy Manhattan step toward the target position."""
        best_directions = []
        min_distance = float("inf")

        for direction in range(4):
            dx, dy = DIRS[direction]
            next_pos = (agent_pos[0] + dx, agent_pos[1] + dy)

            if self.model._cell_blocked(next_pos, small_boxes, heavy_box):
                continue

            distance = abs(next_pos[0] - target_pos[0]) + abs(next_pos[1] - target_pos[1])

            if distance < min_distance:
                min_distance = distance
                best_directions = [direction]
            elif distance == min_distance:
                best_directions.append(direction)

        if not best_directions:
            return self.rng.randrange(4)

        return self.rng.choice(best_directions)
