import gym
import matplotlib.pyplot as plt
import numpy as np

from harl.envs.overcooked.overcooked_game.gym_cooking.environment import cooking_zoo
from harl.envs.overcooked.overcooked_game.gym_cooking.environment.game.graphic_pipeline import (
    GraphicPipeline,
)
from harl.envs.overcooked.overcooked_game.overcooked_wrappers import (
    SARDConsistencyChecker,
)
from harl.utils.lipo_utils import Arrdict, Dotdict, arrdict


class OvercookedMaker:
    def __init__(
        self,
        *,
        mode,
        horizon,
        recipes,
        obs_spaces,
        num_agents=2,
        interact_reward=0.5,
        progress_reward=1.0,
        complete_reward=10.0,
        step_cost=0.1,
        **kwargs
    ):
        if not isinstance(obs_spaces, list):
            obs_spaces = [obs_spaces]
        self._env = cooking_zoo.parallel_env(
            level=mode,
            num_agents=num_agents,
            record=False,
            max_steps=horizon,
            recipes=recipes,
            obs_spaces=obs_spaces,
            interact_reward=interact_reward,
            progress_reward=progress_reward,
            complete_reward=complete_reward,
            step_cost=step_cost,
        )

        self.players = self._env.possible_agents
        self.action_spaces = self._env.action_space(self.players[0])
        self.observation_spaces = Dotdict(
            (k, Dotdict(obs=self._env.observation_space(k))) for k in self.players
        )
        # print('a:',self.action_spaces)
        # print('o:',self.observation_spaces)
        self.graphic_pipeline = GraphicPipeline(
            self._env, display=False
        )  # do not create a display window
        self.graphic_pipeline.on_init()

    def get_action_space(self):
        return gym.spaces.Discrete(6)

    def get_observation_space(self):
        # agent observation size
        # if isinstance(self._env.unwrapped.obs_size, int):
        #    return Dotdict(obs=gym.spaces.Box(-1,1,shape=self._env.unwrapped.obs_size))
        # else:
        #    return Dotdict(obs=gym.spaces.Box(0,10,shape=self._env.unwrapped.obs_size))
        return Dotdict(obs=gym.spaces.Box(-1, 1, shape=self._env.unwrapped.obs_size))
        # return Dotdict(obs=gym.spaces.Box(-1, 1, shape=(59,)))

    def get_state_space(self):
        # agent observation size
        # if isinstance(self._env.unwrapped.obs_size, int):
        #    return Dotdict(obs=gym.spaces.Box(-1,1,shape=self._env.unwrapped.obs_size))
        # else:
        #    return Dotdict(obs=gym.spaces.Box(0,10,shape=self._env.unwrapped.obs_size))
        return Dotdict(obs=gym.spaces.Box(-1, 1, shape=self._env.unwrapped.obs_size))

    def reset(self, seed=None):
        obs = self._env.reset(seed=seed)
        data = Arrdict()
        for p, k in zip(self.players, obs):
            data[p] = Arrdict(obs=obs[k], reward=np.float32(0), done=False)
        return data

    def step(self, decision):
        #        actions = {}
        #        for a,p in zip(self._env.agents,decision.action):
        #            actions[a] = decision.action[p]

        #        obs, reward, done, info = self._env.step(actions)
        # obs, share_obs, reward, done, info = self._env.step(decision)
        obs, reward, done, done2, info = self._env.step(decision)
        data = Arrdict()
        for k in obs.keys():
            data[k] = Arrdict(obs=obs[k], reward=np.float32(reward[k]), done=done[k])
        #        data, info = self._env.step(decision)
        return Arrdict(data), Dotdict(info)

    def render(self, mode):
        return self.graphic_pipeline.on_render(mode)

    @staticmethod
    def make_env(*args, **kwargs):
        env = OvercookedMaker(*args, **kwargs)
        env = SARDConsistencyChecker(env)
        return env


if __name__ == "__main__":
    import argparse

    from coop_marl.controllers import RandomController
    from coop_marl.runners import StepsRunner

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="simple")
    args = parser.parse_args()

    level = "full_divider_salad_4"
    horizon = 200
    recipes = [
        "LettuceSalad",
        "TomatoSalad",
        "ChoppedCarrot",
        "ChoppedOnion",
        "TomatoLettuceSalad",
        "TomatoCarrotSalad",
    ]

    env = OvercookedMaker.make_env(
        obs_spaces="dense", mode=level, horizon=horizon, recipes=recipes
    )
    action_spaces = env.action_spaces
    controller = RandomController(action_spaces)
    runner = StepsRunner(env, controller)
    buffer = []
    for i in range(2):
        traj, infos, frames = runner.rollout(100, render=True)
        print(traj.data)
        buffer.append(traj)
    batch = arrdict.cat(buffer)
    plt.imshow(frames[0])
    plt.show()
