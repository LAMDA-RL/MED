import copy

import numpy as np
from gym.spaces import Discrete


class MatrixGameEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)

        self.n_agents = 2
        self.n_solutions = self.args["n_solutions"]
        self.solution_size = self.parse_solution_size(self.args["solution_size"])
        self.solution_reward = self.parse_solution_reward(self.args["solution_reward"])
        self.n_actions = sum(self.solution_size)
        self.avail_actions = np.ones(self.n_actions)

        self.share_observation_space = [[1], [1]]
        self.observation_space = [[1], [1]]
        self.action_space = [Discrete(self.n_actions), Discrete(self.n_actions)]

        self.matrix = np.zeros([self.n_actions, self.n_actions], dtype=np.float32)
        for m in range(self.n_solutions):
            start = sum(self.solution_size[:m])
            end = sum(self.solution_size[: m + 1])
            self.matrix[start:end, start:end] = self.solution_reward[m]

        self.horizon = self.args["horizon"]
        self.t = 0

    def step(self, actions):
        reward = [
            [self.matrix[actions[0][0]][actions[1][0]]],
            [self.matrix[actions[0][0]][actions[1][0]]],
        ]

        self.t += 1
        done = 1 if self.t >= self.horizon else 0
        dones = [done, done]

        info = [
            {"optimal": reward == np.max(self.matrix)},
            {"optimal": reward == np.max(self.matrix)},
        ]
        next_state = self.get_state()
        next_obs = self.get_obs()
        next_avail_actions = self.get_avail_actions()
        # print(reward)
        return next_obs, next_state, reward, dones, info, next_avail_actions
        # return obs, state, rewards, dones, info, available_actions

    def reset(self):
        # return obs, state, available_actions
        self.t = 0
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def seed(self, seed):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def get_state(self):
        return [[1], [1]]

    def get_obs(self):
        return [[1], [1]]

    def get_avail_actions(self):
        return [self.avail_actions, self.avail_actions]

    def parse_solution_size(self, solution_size):
        if isinstance(solution_size, list):
            solution_size_list = solution_size
        elif isinstance(solution_size, int):
            solution_size_list = [solution_size] * self.n_solutions
        assert len(solution_size_list) == self.n_solutions
        return solution_size_list

    def parse_solution_reward(self, solution_reward):
        if isinstance(solution_reward, str):
            solution_reward_list = eval(solution_reward)
        elif isinstance(solution_reward, list):
            solution_reward_list = solution_reward
        elif isinstance(solution_reward, int):
            solution_reward_list = [solution_reward] * self.n_solutions
        assert len(solution_reward_list) == self.n_solutions
        return solution_reward_list

    def print_matrix(self):
        for line in self.matrix:
            print(line)
