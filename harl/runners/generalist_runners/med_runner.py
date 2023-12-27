import os
import random
import time
from types import SimpleNamespace

import numpy as np
import setproctitle
import torch as th
import torch.nn as nn
from pyinstrument import Profiler
from torch.distributions import Categorical

from harl.algorithms.actors import ALGO_REGISTRY
from harl.common.buffers.trajectory_buffer import TrajectoryBuffer
from harl.common.med_logger import MEDLogger
from harl.utils.action_selectors import MultinomialActionSelector, OneHot
from harl.utils.configs_tools import LogWriter, get_task_name
from harl.utils.envs_tools import (
    get_num_agents,
    get_shape_from_act_space,
    get_shape_from_obs_space,
    make_eval_env,
    make_render_env,
    make_train_env,
    set_seed,
)
from harl.utils.models_tools import init_device
from harl.utils.trans_tools import _t2n


class mg_expert:
    def __init__(self, index, action_num, solution_size):
        self.index = index
        self.action_num = action_num
        self.solution_size = solution_size
        action_begin = 0
        if index >= 1:
            action_begin = sum(solution_size[:index])
        action_end = action_begin + solution_size[index]
        self.solution_actions = list(range(action_begin, action_end))
        self.prob = th.zeros(action_num + 1)
        for i in self.solution_actions:
            self.prob[i] = 1
        self.prob = self.prob / solution_size[index]

    def select_action(self, obs, avail_actions, test_mode):
        probs = np.zeros(avail_actions.shape)
        probs += self.prob[:-1].numpy()
        actions = np.ones([len(avail_actions), 2], dtype=int) * self.index
        actions = np.random.choice(
            self.solution_actions, [len(avail_actions), 2], replace=True
        )
        return actions

    def get_target(self, targets):
        targets = th.nn.functional.one_hot(
            targets.to(th.int64), self.action_num + 1
        ).float()
        return targets


class lipo_expert:
    def __init__(self, index, action_num, agent, device):
        self.index = index
        self.action_num = action_num
        self.agent = agent
        self.device = device
        self.reset()

    def select_action(self, obs, avail_actions, test_mode):
        a_dist = self.agent.calc_action_dist(th.Tensor(obs), th.Tensor(self.z))
        actions = a_dist.sample().detach().cpu().numpy()
        return actions

    def reset(self):
        self.z = []
        for i in range(8):
            self.z.append(self.sample_z().copy())
        self.z = np.array(self.z)

    def sample_z(self):
        z_dim = 8
        if isinstance(z_dim, int):
            z = random.choices(np.eye(z_dim, dtype=np.float32), k=2)
        return np.array(z)

    def get_target(self, targets):
        targets = th.nn.functional.one_hot(
            targets.to(th.int64), self.action_num + 1
        ).float()
        return targets


class MEDRunner:
    def __init__(self, args, algo_args, env_args):
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        print("args:")
        print(self.args)
        print("algo_args:")
        print(self.algo_args)
        print("env_args:")
        print(self.env_args)

        self.device = init_device(algo_args["device"])
        self.n_rollout_threads = {
            "train": algo_args["train"]["n_rollout_threads"],
            "eval": algo_args["eval"]["n_eval_rollout_threads"],
        }
        self.n_episodes = self.algo_args["algo"]["n_episodes"]

        self.envs = make_train_env(
            args["env"],
            algo_args["seed"]["seed"],
            self.n_rollout_threads["train"],
            env_args,
        )
        self.eval_envs = (
            make_eval_env(
                args["env"],
                algo_args["seed"]["seed"],
                self.n_rollout_threads["eval"],
                env_args,
            )
            if algo_args["eval"]["use_eval"]
            else None
        )
        self.n_agents = get_num_agents(args["env"], env_args, self.envs)
        self.total_agents = self.n_agents
        self.n_trained_agents = self.algo_args["algo"]["n_trained_agents"]
        self.n_expert_controlled_agents = self.total_agents - self.n_trained_agents

        self.episode_limit = self.env_args["horizon"]
        self.algo_args["train"]["episode_length"] = self.episode_limit

        self.state_space = self.envs.share_observation_space
        self.obs_space = self.envs.observation_space
        self.action_space = self.envs.action_space
        self.state_shape = get_shape_from_obs_space(self.state_space[0])
        self.obs_shape = get_shape_from_obs_space(self.obs_space[0])
        self.action_shape = get_shape_from_act_space(self.action_space[0])
        self.n_actions = self.action_space[0].n
        self.vocab_size = self.n_actions
        self.transition_dim = 3

        self.gpt_config = SimpleNamespace(**self.algo_args["model"])
        self.gpt_config.vocab_size = self.n_actions
        self.gpt_config.transition_dim = 3
        self.gpt_config.block_size = (
            self.episode_limit + 1
        ) * self.gpt_config.transition_dim * self.n_episodes + 1
        self.gpt_config.obs_shape = np.prod(self.obs_shape[0])
        self.gpt_config.action_shape = self.action_shape
        self.agent = ALGO_REGISTRY[args["algo"]](self.gpt_config).to(self.device)

        self.action_selector = MultinomialActionSelector()
        if self.args["env"] == "matrix_game":
            expert_index_lst = list(range(self.env_args["n_solutions"]))
            self.n_expert = len(expert_index_lst)
            self.expert_pool = [
                mg_expert(i, self.n_actions, self.env_args["solution_size"])
                for i in expert_index_lst
            ]
        elif self.args["env"] == "overcooked":
            self.population = th.load(
                "../harl/runners/generalist_runners/models/overcooked_population.pt"
            )
            self.n_expert = len(self.population)
            self.expert_pool = [
                lipo_expert(i, self.n_actions, self.population[i], self.device)
                for i in range(self.n_expert)
            ]
        elif self.args["env"] == "gridworld":
            self.population = th.load(
                "../harl/runners/generalist_runners/models/movebox_population.pt"
            )
            self.n_expert = len(self.population)
            self.expert_pool = [
                lipo_expert(i, self.n_actions, self.population[i], self.device)
                for i in range(self.n_expert)
            ]
        else:
            print("Can not support the " + self.args["env"] + "environment.")
            raise NotImplementedError

        # Set up buffer for rollout

        self.scheme = {
            "state": {"vshape": self.state_shape, "dtype": th.float},
            "obs": {
                "vshape": self.obs_shape,
                "dtype": th.float,
                "group": self.total_agents,
            },
            "avail_actions": {
                "vshape": self.n_actions,
                "dtype": th.int,
                "group": self.total_agents,
            },
            "actions": {
                "vshape": 1,
                "dtype": th.int,
                "group": self.total_agents,
                "preprocess": ("actions_onehot", [OneHot(out_dim=self.n_actions)]),
            },
            "target_actions": {
                "vshape": 1,
                "dtype": th.int,
                "group": self.n_trained_agents,
            },
            "reward": {"vshape": 1, "dtype": th.float},
            "done": {"vshape": 1, "dtype": th.uint8},
        }

        self.train_buffer = TrajectoryBuffer(
            self.scheme,
            self.algo_args["train"]["buffer_size"] * self.n_expert,
            0,
            (self.episode_limit + 1) * self.n_episodes,
            vocab=self.n_actions,
            game=self.args["env"],
            device="cpu",
        )
        self.eval_buffer = [
            TrajectoryBuffer(
                self.scheme,
                self.algo_args["train"]["buffer_size"],
                buffer_index,
                (self.episode_limit + 1) * self.n_episodes,
                vocab=self.n_actions,
                game=self.args["env"],
                device="cpu",
            )
            for buffer_index in range(self.n_expert)
        ]
        self.accuracy_lst = [0] * self.n_episodes
        self.t_env = 0

        self.writer = LogWriter(args, algo_args, env_args)
        self.run_dir, self.log_dir, self.save_dir = self.writer.get_dirs()

        self.logger = MEDLogger(
            args, algo_args, env_args, self.n_agents, self.writer, self.run_dir
        )

        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )
        
        self.profiler = Profiler(interval=self.algo_args["logger"]["profiler_interval"])
        self.profiler.start()

    def run(self):
        print("run")
        epoch = 0

        self.logger.init(self.algo_args["train"]["t_max"])
        while self.t_env < self.algo_args["train"]["t_max"]:
            epoch += 1
            self.logger.episode_init(
                epoch, self.t_env
            )  # logger callback at the beginning of each episode

            optimizer = self.agent.configure_optimizers(self.gpt_config)
            # sample
            if (
                self.train_buffer.trajectories_in_buffer
                + self.algo_args["train"]["batch_size"]
                >= self.algo_args["train"]["buffer_size"] * self.n_expert
            ):
                self.train_buffer.empty()
            with th.no_grad():
                for expert_index in range(self.n_expert):
                    self.rollout(expert_index, tag="train")

            # train
            for i in range(self.n_expert):
                (
                    obs_batch,
                    action_batch,
                    reward_batch,
                    mask,
                    targets,
                ) = self.train_buffer.sample(self.algo_args["train"]["batch_size"], 0)
                if self.args["env"] == "matrix_game":
                    obs_batch = obs_batch.to(th.float32)
                    reward_batch = reward_batch.to(th.float32)
                targets = self.expert_pool[expert_index].get_target(
                    th.reshape(targets, (self.algo_args["train"]["batch_size"], -1))
                )
                transformer_output, loss = self.agent(
                    obs_batch.to(self.device),
                    action_batch.to(self.device),
                    reward_batch.to(self.device),
                    targets=targets.reshape(-1, targets.size(-1)).to(self.device),
                    mask=mask.to(self.device),
                )

                self.agent.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.algo_args["model"]["grad_norm_clip"]
                )
                optimizer.step()
            self.logger.episode_log(self.t_env, loss.item())
            # test
            if (
                epoch % self.algo_args["eval"]["eval_interval"] == 0
            ) or self.t_env >= self.algo_args["train"]["t_max"]:
                eval_reward = [0] * self.n_episodes
                self.accuracy_lst = [0] * self.n_episodes
                for expert_index in range(self.n_expert):
                    self.eval_buffer[expert_index].empty()
                    with th.no_grad():
                        self.rollout(expert_index, tag="eval")
                    eval_reward_lst = self.eval_buffer[expert_index].get_eval_reward()
                    for i, r in enumerate(eval_reward_lst):
                        eval_reward[i] += r / self.n_rollout_threads["eval"]
                self.logger.eval_log(
                    eval_reward,
                    self.n_expert,
                    self.t_env,
                    accuracy_lst=self.accuracy_lst,
                )
        self.save()
        self.envs.close()

    def rollout(self, expert_index, tag="train"):
        # sample (n_rollout_threads) cross-k-episode trajectories
        test_mode = tag == "eval"
        if test_mode:
            buffer = self.eval_buffer[expert_index]
        else:
            buffer = self.train_buffer
        expert = self.expert_pool[expert_index]

        t = [0 for _ in range(self.n_rollout_threads[tag])]
        accuracy_cnt = [0] * self.n_episodes
        total_cnt = [0] * self.n_episodes
        for tn in range(self.n_episodes):
            obs, states, avail_actions = self.envs.reset()

            envs_not_done = list(range(self.n_rollout_threads[tag]))
            initial_data = {
                "state": states[:, 0],
                "obs": obs,
                "avail_actions": avail_actions,
            }
            buffer.update(initial_data, bs=envs_not_done, ts=t, mark_filled=False)

            while True:
                obs_input_expert, avail_actions_input = buffer.get_expert_input(
                    self.n_rollout_threads[tag], envs_not_done, self.n_trained_agents, t
                )
                a_expert = expert.select_action(
                    obs_input_expert,
                    avail_actions_input,
                    test_mode=True,
                )
                a_target = a_expert[envs_not_done, : self.n_trained_agents].copy()[:, 0]
                a_expert = a_expert[envs_not_done, self.n_trained_agents :].copy()

                # Get input actions
                (
                    obs_input_transformer,
                    action_input_transformer,
                    reward_input_transformer,
                ) = buffer.get_transformer_input(envs_not_done, 0, t)
                if self.args["env"] == "matrix_game":
                    obs_input_transformer = obs_input_transformer.to(th.float32)
                    reward_input_transformer = reward_input_transformer.to(th.float32)
                transformer_output, _ = self.agent(
                    obs_input_transformer.to(self.device),
                    action_input_transformer.to(self.device),
                    reward_input_transformer.to(self.device),
                )

                prob = []
                for i, e in enumerate(envs_not_done):
                    prob.append(transformer_output[i, t[e] * 3])
                prob = th.stack(prob)
                avail_actions_input_transformer = th.Tensor(
                    np.hstack(
                        [
                            avail_actions_input[envs_not_done, 0],
                            np.zeros([len(envs_not_done), 1]),
                        ]
                    )
                )
                a_transformer = (
                    self.action_selector.select_action(
                        prob, avail_actions_input_transformer, t, test_mode=True
                    )
                    .cpu()
                    .numpy()
                )
                if len(a_expert.shape) == 1:
                    a_expert = np.expand_dims(a_expert, axis=1)
                if len(a_transformer.shape) == 1:
                    a_transformer = np.expand_dims(a_transformer, axis=1)

                accuracy_cnt[tn] += np.sum(a_target == a_transformer[:, 0])
                total_cnt[tn] += len(envs_not_done)

                actions = np.hstack([a_transformer, a_expert])
                action_data = {
                    "actions": th.Tensor(actions).unsqueeze(1),
                    "target_actions": th.Tensor(a_target).unsqueeze(1),
                }
                if len(actions.shape) == 2:
                    actions = np.expand_dims(actions, axis=2)
                buffer.update(action_data, bs=envs_not_done, ts=t, mark_filled=False)

                full_actions = np.zeros(
                    [self.n_rollout_threads[tag], self.total_agents, actions.shape[-1]],
                    dtype=int,
                )
                for i, e in enumerate(envs_not_done):
                    full_actions[e] = actions[i].copy()
                obs, state, reward, done, info, avail_actions = self.envs.step(
                    full_actions
                )
                obs = obs[envs_not_done]
                state = state[envs_not_done]
                reward = reward[envs_not_done]
                done = done[envs_not_done]
                avail_actions = avail_actions[envs_not_done]
                # Get env returns
                this_timestep_data = {
                    "reward": reward[:, 0],
                    "done": done[:, 0],
                }
                buffer.update(
                    this_timestep_data, bs=envs_not_done, ts=t, mark_filled=False
                )
                if not test_mode:
                    self.logger.per_step(this_timestep_data, envs_not_done)
                # Data for the next step we will insert in order to select an action
                next_timestep_data = {
                    "state": state[:, 0],
                    "obs": obs,
                    "avail_actions": avail_actions,
                }
                for ti in envs_not_done:
                    t[ti] += 1
                buffer.update(
                    next_timestep_data, bs=envs_not_done, ts=t, mark_filled=True
                )

                new_envs_not_done = []
                for i, env_id in enumerate(envs_not_done):
                    if not done[i, 0]:
                        new_envs_not_done.append(env_id)

                # Update envs_not_done
                envs_not_done = new_envs_not_done
                if not envs_not_done:
                    buffer.update_episode_finish_info(t)
                    break

        buffer.update_finish_info(t)
        sample_stats = {"n_episodes": self.n_rollout_threads[tag] * self.n_episodes}

        if test_mode:
            for i in range(self.n_episodes):
                self.accuracy_lst[i] += accuracy_cnt[i] / total_cnt[i]
        else:
            self.t_env += sum(total_cnt)

    def update_stats(self, stats, tag):
        self.stats[tag] = self.stats.get(tag, {})
        for k, v in stats.items():
            self.stats[tag][k] = self.stats[tag].get(k, 0) + v

    def save(self):
        print("save to:" + str(self.save_dir))
        th.save(self.agent, str(self.save_dir) + "/med.pt")

    def close(self):
        print("close")
        self.logger.close()
        self.envs.close()
        self.profiler.stop()
        with open(os.path.join(self.log_dir, "profile.html"), "w") as f:
            f.write(self.profiler.output_html())
