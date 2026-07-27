import time
import numpy as np
import random
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from exercises.ex4.pomdp import GenerativeModel, MultiAgentGenerativeModel, run_online_planning


class SingleAgentPOMDPWrapper:
    def __init__(self, env, generative_model, agent_name="agent_0"):
        self.env = env
        self.gen_model = generative_model
        self.agent_name = agent_name

    def _get_true_state(self):
        pos = self.env.agent_positions[self.agent_name]
        d = self.env.agent_dirs[self.agent_name]

        s_boxes, h_boxes = [], []
        for y in range(self.env.height):
            for x in range(self.env.width):
                cell = self.env.core_env.grid.get(x, y)
                if cell is not None and cell.type == "box":
                    if getattr(cell, "box_size", "") == "heavy":
                        h_boxes.append((x, y))
                    else:
                        s_boxes.append((x, y))

        return (pos[0], pos[1], d, tuple(s_boxes), tuple(h_boxes))

    def reset(self):
        self.env.reset()
        true_state = self._get_true_state()
        obs = self.gen_model.get_obs(true_state)
        return obs, {}

    def step(self, action):
        _, rewards, terms, truncs, infos = self.env.step({self.agent_name: action})

        done = not self.env.agents
        if done:
            return None, 1.0, True, False, {}

        true_state = self._get_true_state()
        obs = self.gen_model.get_obs(true_state)

        return obs, rewards.get(self.agent_name, 0.0), terms.get(self.agent_name, False), truncs.get(self.agent_name,
                                                                                                     False), infos.get(
            self.agent_name, {})


class MultiAgentPOMDPWrapper:
    def __init__(self, env, generative_model):
        self.env = env
        self.gen_model = generative_model
        self.agent_names = ["agent_0", "agent_1"]

    def _get_true_state(self):
        s_boxes, h_boxes = [], []
        for y in range(self.env.height):
            for x in range(self.env.width):
                cell = self.env.core_env.grid.get(x, y)
                if cell is not None and cell.type == "box":
                    if getattr(cell, "box_size", "") == "heavy":
                        h_boxes.append((x, y))
                    else:
                        s_boxes.append((x, y))

        p0 = self.env.agent_positions[self.agent_names[0]]
        p1 = self.env.agent_positions[self.agent_names[1]]
        d0 = self.env.agent_dirs[self.agent_names[0]]
        d1 = self.env.agent_dirs[self.agent_names[1]]
        return ((p0[0], p0[1], d0), (p1[0], p1[1], d1), tuple(s_boxes), tuple(h_boxes))

    def reset(self):
        self.env.reset()
        return self.gen_model.get_obs(self._get_true_state()), {}

    def step(self, joint_action):
        action_dict = {
            self.agent_names[0]: joint_action // 3,
            self.agent_names[1]: joint_action % 3
        }
        _, rewards, terms, truncs, infos = self.env.step(action_dict)

        if not self.env.agents:
            return None, 1.0, True, False, {}

        obs = self.gen_model.get_obs(self._get_true_state())
        return obs, sum(rewards.values()), all(terms.values()), all(truncs.values()), {}


def parse_ascii_map(ascii_map):
    walls, goals, s_boxes, h_boxes, free_cells = [], [], [], [], []
    for y, row in enumerate(ascii_map):
        for x, char in enumerate(row):
            if char == 'W':
                walls.append((x, y))
            elif char == 'G':
                goals.append((x, y))
                free_cells.append((x, y))
            elif char == 'B':
                s_boxes.append((x, y))
                free_cells.append((x, y))
            elif char == 'C':
                h_boxes.append((x, y))
                free_cells.append((x, y))
            elif char == 'A' or char == ' ':
                free_cells.append((x, y))
    return walls, goals, s_boxes, h_boxes, free_cells


def main():
    ascii_map = [
        "WWWWWWW",
        "W A   W",
        "W B   W",
        "W  BG W",
        "W G   W",
        "WWWWWWW"
    ]

    walls, goals, initial_s_boxes, initial_h_boxes, free_cells = parse_ascii_map(ascii_map)
    generative_model = GenerativeModel(walls, goals)

    time_budgets = [1.0, 20.0]
    n_runs = 30
    n_particles = 500

    for budget in time_budgets:
        print(f"--- Running experiments with Time Budget: {budget}s ---")
        steps_recorded = []

        for i in range(n_runs):
            env = StochasticMultiAgentBoxPushEnv(ascii_map=ascii_map, max_steps=100)
            wrapper = SingleAgentPOMDPWrapper(env, generative_model)

            initial_particles = []
            for _ in range(n_particles):
                px, py = random.choice(free_cells)
                pd = random.randint(0, 3)
                p_state = (px, py, pd, tuple(initial_s_boxes), tuple(initial_h_boxes))
                initial_particles.append(p_state)

            steps = run_online_planning(
                env=wrapper, generative_model=generative_model,
                initial_particles=initial_particles, free_cells=free_cells, time_limit=budget
            )

            steps_recorded.append(steps)
            print(f"Run {i + 1}/{n_runs} completed in {steps} steps.")

        print(
            f"\nSingle-Agent Results for {budget}s: Mean Steps = {np.mean(steps_recorded):.2f}, Std = {np.std(steps_recorded):.2f}\n")


def run_multi_agent_experiment():
    ascii_map_two_agents = [
        "WWWWWWWW",
        "W  AA  W",
        "W B C  W",
        "W      W",
        "W      W",
        "W G G  W",
        "WWWWWWWW"
    ]

    walls, goals, s_boxes, h_boxes, free_cells = parse_ascii_map(ascii_map_two_agents)
    generative_model = MultiAgentGenerativeModel(walls, goals)

    time_budgets = [1.0, 20.0]
    n_runs = 30
    n_particles = 1000

    for budget in time_budgets:
        print(f"--- Running MULTI-AGENT experiments with Time Budget: {budget}s ---")
        steps_recorded = []

        for i in range(n_runs):
            env = StochasticMultiAgentBoxPushEnv(ascii_map=ascii_map_two_agents, max_steps=100)
            wrapper = MultiAgentPOMDPWrapper(env, generative_model)

            initial_particles = []
            for _ in range(n_particles):
                px0, py0 = random.choice(free_cells)
                px1, py1 = random.choice(free_cells)
                p_state = ((px0, py0, random.randint(0, 3)), (px1, py1, random.randint(0, 3)), tuple(s_boxes),
                           tuple(h_boxes))
                initial_particles.append(p_state)

            steps = run_online_planning(
                env=wrapper, generative_model=generative_model,
                initial_particles=initial_particles, free_cells=free_cells, time_limit=budget
            )

            steps_recorded.append(steps)
            print(f"Run {i + 1}/{n_runs} completed in {steps} steps.")

        print(
            f"\nMulti-Agent Results for {budget}s: Mean Steps = {np.mean(steps_recorded):.2f}, Std = {np.std(steps_recorded):.2f}\n")


if __name__ == "__main__":
    # print("Starting Single Agent...")
    # main()
    print("\nStarting Multi-Agent...")
    run_multi_agent_experiment()