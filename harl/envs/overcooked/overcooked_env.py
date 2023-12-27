import copy
import gc
import os

import numpy as np
from gym.spaces import Discrete

from harl.envs.overcooked.overcooked_game.overcooked_maker import OvercookedMaker
from harl.envs.overcooked.overcooked_game.overcooked_wrappers import wrap
from harl.utils.lipo_utils import save_gif


class OvercookedEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)

        self.n_agents = args["num_agents"]
        self.n_actions = 6
        self.avail_actions = np.ones(self.n_actions)
        self.avail_actions = np.ones(self.n_actions)
        self.unwrapped_env = OvercookedMaker.make_env(
            obs_spaces=self.args["obs_spaces_type"],
            mode=self.args["map_name"],
            horizon=self.args["horizon"],
            recipes=self.args["recipes"],
            num_agents=self.n_agents,
            interact_reward=self.args["interact_reward"],
            progress_reward=self.args["progress_reward"],
            complete_reward=self.args["complete_reward"],
            step_cost=self.args["step_cost"],
        )
        self.players = self.unwrapped_env.players
        assert self.n_agents == len(self.players)
        self.env = wrap(
            env=self.unwrapped_env,
            wrappers=self.args["env_wrappers"],
            **self.args["wrapper_args"]
        )
        self.share_observation_space = [
            self.env.get_state_space()["obs"] for i in range(self.n_agents)
        ]
        self.observation_space = [
            self.env.get_observation_space()["obs"] for i in range(self.n_agents)
        ]
        self.action_space = [self.env.get_action_space() for i in range(self.n_agents)]

        self.frames = []
        self.render_mode = "rgb_array"

    def step(self, actions):
        decision = {}
        for i, p in enumerate(self.players):
            decision[p] = actions[i][0]
        data, info = self.env.step(decision)
        obs = [data[p]["obs"] for p in self.players]
        state = [data[p]["state"] for p in self.players]
        rewards = [[data[p]["reward"]] for p in self.players]
        dones = [1 if data[p]["done"] else 0 for p in self.players]
        available_actions = [self.avail_actions for p in self.players]

        frame = self.env.render(mode=self.render_mode)
        self.frames.append(frame)

        return obs, state, rewards, dones, [{}, {}], available_actions

    def reset(self):
        data = self.env.reset(seed=self.seed)
        obs = [data[p]["obs"] for p in self.players]
        state = [data[p]["state"] for p in self.players]
        avail_actions = [self.avail_actions for p in self.players]

        self.frames = []
        frame = self.env.render(mode=self.render_mode)
        self.frames.append(frame)

        return obs, state, avail_actions

    def seed(self, seed):
        self.seed = seed

    def save_replay(self, name="replay"):
        path = os.path.join("./results", "replays", "overcooked", name + ".gif")
        save_gif(self.frames, path, fps=20, size=(200, 200))
        self.frames = []
        gc.collect()

    # use save_replay instead
    def render(self):
        pass

    def close(self):
        pass
