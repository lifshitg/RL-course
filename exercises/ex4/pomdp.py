import time
import math
import random


class GenerativeModel:
    def __init__(self, walls, goals):
        self.walls = set(walls)
        self.goals = set(goals)
        # 0: left, 1: right, 2: forward
        self.actions = [0, 1, 2]
        # 0: right, 1: down, 2: left, 3: up
        self.dir_to_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def get_actions(self):
        return self.actions

    def step(self, state, action):
        # state representation: (x, y, direction, small_boxes_tuple, heavy_boxes_tuple)
        x, y, d, s_boxes, h_boxes = state

        if action == 0:
            next_state = (x, y, (d - 1) % 4, s_boxes, h_boxes)
            return next_state, self.get_obs(next_state), 0.0

        if action == 1:
            next_state = (x, y, (d + 1) % 4, s_boxes, h_boxes)
            return next_state, self.get_obs(next_state), 0.0

        vec = self.dir_to_vec[d]
        fwd_pos = (x + vec[0], y + vec[1])

        next_x, next_y = x, y
        next_s_boxes = list(s_boxes)

        is_push = fwd_pos in s_boxes or fwd_pos in h_boxes

        if is_push:
            if random.random() < 0.8:
                if fwd_pos in s_boxes:
                    fwd_fwd_pos = (fwd_pos[0] + vec[0], fwd_pos[1] + vec[1])
                    if self._is_free(fwd_fwd_pos, s_boxes, h_boxes):
                        next_s_boxes.remove(fwd_pos)
                        next_s_boxes.append(fwd_fwd_pos)
                        next_x, next_y = fwd_pos
        else:
            r = random.random()
            actual_d = d
            if r > 0.8:
                actual_d = (d - 1) % 4 if r < 0.9 else (d + 1) % 4

            a_vec = self.dir_to_vec[actual_d]
            a_fwd = (x + a_vec[0], y + a_vec[1])

            if self._is_free(a_fwd, s_boxes, h_boxes):
                next_x, next_y = a_fwd

        next_state = (next_x, next_y, d, tuple(next_s_boxes), h_boxes)
        reward = 1.0 if self._check_win(next_s_boxes, h_boxes) else 0.0
        return next_state, self.get_obs(next_state), reward

    def _is_free(self, pos, s_boxes, h_boxes):
        return pos not in self.walls and pos not in s_boxes and pos not in h_boxes

    def _check_win(self, s_boxes, h_boxes):
        covered = sum(1 for g in self.goals if g in s_boxes or g in h_boxes)
        return covered == len(self.goals)

    def get_obs(self, state):
        x, y, _, s_boxes, h_boxes = state
        obs = []

        for i in range(x - 1, x + 2):
            row = []
            for j in range(y - 1, y + 2):
                pos = (i, j)
                if pos in self.walls:
                    row.append('WALL')
                elif pos in s_boxes:
                    row.append('SMALL_BOX')
                elif pos in h_boxes:
                    row.append('HEAVY_BOX')
                elif pos in self.goals:
                    row.append('GOAL')
                else:
                    row.append('EMPTY')
            obs.append(tuple(row))

        return tuple(obs)


class MultiAgentGenerativeModel(GenerativeModel):
    def __init__(self, walls, goals):
        super().__init__(walls, goals)
        # Base-3 encoding for 9 joint actions
        self.actions = list(range(9))

    def get_obs(self, state):
        p0, p1, s_boxes, h_boxes = state
        obs_0 = super().get_obs((p0[0], p0[1], p0[2], s_boxes, h_boxes))
        obs_1 = super().get_obs((p1[0], p1[1], p1[2], s_boxes, h_boxes))
        return (obs_0, obs_1)

    def _resolve_single_agent(self, x, y, d, action, s_boxes, h_boxes):
        if action == 0: return (x, y, (d - 1) % 4), list(s_boxes)
        if action == 1: return (x, y, (d + 1) % 4), list(s_boxes)

        vec = self.dir_to_vec[d]
        fwd_pos = (x + vec[0], y + vec[1])
        next_x, next_y = x, y
        next_s_boxes = list(s_boxes)

        if fwd_pos in s_boxes:
            if random.random() < 0.8:
                fwd_fwd_pos = (fwd_pos[0] + vec[0], fwd_pos[1] + vec[1])
                if self._is_free(fwd_fwd_pos, s_boxes, h_boxes):
                    next_s_boxes.remove(fwd_pos)
                    next_s_boxes.append(fwd_fwd_pos)
                    next_x, next_y = fwd_pos
        elif not (fwd_pos in h_boxes):
            r = random.random()
            actual_d = d
            if r > 0.8:
                actual_d = (d - 1) % 4 if r < 0.9 else (d + 1) % 4
            a_vec = self.dir_to_vec[actual_d]
            a_fwd = (x + a_vec[0], y + a_vec[1])
            if self._is_free(a_fwd, s_boxes, h_boxes):
                next_x, next_y = a_fwd

        return (next_x, next_y, d), next_s_boxes

    def step(self, state, action):
        p0, p1, s_boxes, h_boxes = state
        a0, a1 = action // 3, action % 3

        next_h_boxes = list(h_boxes)

        # Joint heavy box push
        if a0 == 2 and a1 == 2 and p0 == p1:
            vec = self.dir_to_vec[p0[2]]
            fwd_pos = (p0[0] + vec[0], p0[1] + vec[1])
            if fwd_pos in h_boxes:
                if random.random() < 0.8:
                    fwd_fwd_pos = (fwd_pos[0] + vec[0], fwd_pos[1] + vec[1])
                    if self._is_free(fwd_fwd_pos, s_boxes, h_boxes):
                        next_h_boxes.remove(fwd_pos)
                        next_h_boxes.append(fwd_fwd_pos)
                        p0 = p1 = (fwd_pos[0], fwd_pos[1], p0[2])

                next_state = (p0, p1, tuple(s_boxes), tuple(next_h_boxes))
                reward = 1.0 if self._check_win(s_boxes, next_h_boxes) else 0.0
                return next_state, self.get_obs(next_state), reward

        next_p0, next_s_boxes_0 = self._resolve_single_agent(p0[0], p0[1], p0[2], a0, s_boxes, h_boxes)
        next_p1, next_s_boxes_1 = self._resolve_single_agent(p1[0], p1[1], p1[2], a1, next_s_boxes_0, h_boxes)

        next_state = (next_p0, next_p1, tuple(next_s_boxes_1), tuple(next_h_boxes))
        reward = 1.0 if self._check_win(next_s_boxes_1, next_h_boxes) else 0.0

        return next_state, self.get_obs(next_state), reward


class ParticleFilter:
    def __init__(self, initial_particles):
        self.n_particles = len(initial_particles)
        self.particles = initial_particles

    def update(self, action, real_obs, generative_model, free_cells):
        new_particles = []
        max_attempts = self.n_particles * 100
        attempts = 0

        while len(new_particles) < self.n_particles and attempts < max_attempts:
            state = random.choice(self.particles)
            next_state, sim_obs, _ = generative_model.step(state, action)

            if sim_obs == real_obs:
                new_particles.append(next_state)
            attempts += 1

        if not new_particles:
            is_multi_agent = isinstance(self.particles[0][0], tuple)

            for _ in range(self.n_particles):
                if is_multi_agent:
                    p0 = random.choice(free_cells)
                    p1 = random.choice(free_cells)
                    dummy_state = (
                        (p0[0], p0[1], random.randint(0, 3)),
                        (p1[0], p1[1], random.randint(0, 3)),
                        tuple(),
                        tuple()
                    )
                else:
                    cell = random.choice(free_cells)
                    dummy_state = (cell[0], cell[1], random.randint(0, 3), tuple(), tuple())

                if generative_model.get_obs(dummy_state) == real_obs:
                    new_particles.append(dummy_state)

            if not new_particles:
                new_particles = self.particles.copy()

        if new_particles:
            self.particles = random.choices(new_particles, k=self.n_particles)
        else:
            self.particles = self.particles.copy()


class POMCPNode:
    def __init__(self):
        self.visits = 0
        self.value = 0.0
        self.children = {}


class POMCP:
    def __init__(self, generative_model, gamma=0.99, c=1.0, max_depth=30):
        self.model = generative_model
        self.gamma = gamma
        self.c = c
        self.max_depth = max_depth

    def search(self, particles, time_budget):
        root = POMCPNode()
        start_time = time.time()

        while time.time() - start_time < time_budget:
            state = random.choice(particles)
            self.simulate(state, root, depth=0)

        best_action = None
        best_value = -float('inf')

        for action, obs_nodes in root.children.items():
            if not obs_nodes:
                continue
            avg_val = sum(child.value for child in obs_nodes.values()) / len(obs_nodes)
            if avg_val > best_value:
                best_value = avg_val
                best_action = action

        if best_action is None:
            return random.choice(self.model.get_actions())

        return best_action

    def simulate(self, state, node, depth):
        if depth >= self.max_depth:
            return 0.0

        if not node.children:
            for action in self.model.get_actions():
                node.children[action] = {}
            return self.rollout(state, depth)

        best_action = self._ucb1(node)
        next_state, obs, reward = self.model.step(state, best_action)

        if obs not in node.children[best_action]:
            node.children[best_action][obs] = POMCPNode()

        q_value = reward + self.gamma * self.simulate(
            next_state,
            node.children[best_action][obs],
            depth + 1
        )

        node.visits += 1
        node.value += (q_value - node.value) / node.visits
        return q_value

    def _ucb1(self, node):
        best_action = None
        best_ucb = -float('inf')

        for action, obs_nodes in node.children.items():
            action_visits = sum(child.visits for child in obs_nodes.values())
            if action_visits == 0:
                return action

            action_value = sum(child.value for child in obs_nodes.values()) / len(obs_nodes)
            ucb = action_value + self.c * math.sqrt(math.log(node.visits) / action_visits)

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action

    def rollout(self, state, depth):
        curr_state = state
        total_reward = 0.0
        discount = 1.0

        for _ in range(depth, self.max_depth):
            action = random.choice(self.model.get_actions())
            curr_state, _, reward = self.model.step(curr_state, action)
            total_reward += discount * reward
            discount *= self.gamma

        return total_reward


def run_online_planning(env, generative_model, initial_particles, free_cells, time_limit):
    obs, info = env.reset()
    belief = ParticleFilter(initial_particles)
    pomcp_planner = POMCP(generative_model)

    done = False
    total_steps = 0

    while not done:
        action = pomcp_planner.search(belief.particles, time_budget=time_limit)

        obs, reward, terminated, truncated, info = env.step(action)

        belief.update(action, obs, generative_model, free_cells)

        done = terminated or truncated
        total_steps += 1

    return total_steps