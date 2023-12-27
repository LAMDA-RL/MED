import copy
import gc
import os

import numpy as np
from gym.spaces import Discrete, MultiDiscrete

from harl.envs.gridworld.tasks.MoveBox.env_MoveBox import EnvMoveBox
from harl.utils.lipo_utils import save_gif


class GridWorldEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.task = self.args["task"]
        self.n_agents = self.args["n_agents"]
        self.map_name = self.args["map"]
        self.map_size = self.args["map_size"]

        if self.task == "MoveBox":
            assert self.n_agents == 2, "MoveBox only supports 2 agents"
            self.env = EnvMoveBox(
                self.args["horizon"], self.map_name, self.args["obs_type"]
            )
            self.map_size = self.env.map_size
            self.observation_space = [[27] for i in range(self.n_agents)]
        else:
            print("Can not support the " + self.task + "task.")
            raise NotImplementedError
        self.share_observation_space = [
            [self.map_size[0] * self.map_size[1] * 3] for i in range(self.n_agents)
        ]
        self.action_space = [Discrete(4) for i in range(self.n_agents)]
        self.available_actions = [[1, 1, 1, 1] for i in range(self.n_agents)]

        self.frames = []

    def step(self, actions):
        # return obs, state, rewards, dones, info, available_actions
        action_lst = [a[0] for a in actions]
        reward, done = self.env.step(action_lst)
        frame = self.env.render()
        self.frames.append(frame)
        return (
            self.env.get_obs(),
            [
                self.env.get_global_obs(self.args["obs_type"]),
                self.env.get_global_obs(self.args["obs_type"]),
            ],
            [[reward] for i in range(self.n_agents)],
            [1 if done else 0 for i in range(self.n_agents)],
            [{}, {}],
            self.available_actions,
        )

    def reset(self):
        # return obs, state, available_actions
        self.env.reset()
        self.frames = []
        frame = self.env.render()
        self.frames.append(frame)
        return (
            self.env.get_obs(),
            [
                self.env.get_global_obs(self.args["obs_type"]),
                self.env.get_global_obs(self.args["obs_type"]),
            ],
            self.available_actions,
        )

    def seed(self, seed):
        pass

    def save_replay(self, name="replay"):
        path = os.path.join(
            "./results", "replays", "gridworld", self.task, name + ".gif"
        )
        save_gif(self.frames, path, fps=10)
        self.frames = []
        gc.collect()

    def render(self):
        pass

    def close(self):
        pass
