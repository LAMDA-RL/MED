import gc
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict, deque
from functools import partial
from itertools import product

import numpy as np
import psutil
import setproctitle
import torch
from pyinstrument import Profiler

from harl.algorithms.actors import ALGO_REGISTRY
from harl.common.lipo_logger import LIPOLogger
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
from harl.utils.lipo_utils import (
    Arrdict,
    Dotdict,
    arrdict,
    get_avg_metrics,
    get_traj_info,
    merge_dict,
)
from harl.utils.models_tools import init_device
from harl.utils.trans_tools import _t2n


class LIPORunner:
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
            "xp": algo_args["train"]["n_xp_rollout_threads"],
        }

        # environments
        self.envs = make_train_env(
            args["env"],
            algo_args["seed"]["seed"],
            self.n_rollout_threads["train"],
            env_args,
        )
        self.xp_envs = make_train_env(
            args["env"],
            algo_args["seed"]["seed"],
            self.n_rollout_threads["xp"],
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
        self.state_space = self.envs.share_observation_space
        self.obs_space = self.envs.observation_space
        self.action_space = self.envs.action_space
        self.state_shape = get_shape_from_obs_space(self.state_space[0])
        self.obs_shape = get_shape_from_obs_space(self.obs_space[0])
        self.action_shape = get_shape_from_act_space(self.action_space[0])
        self.n_agents = get_num_agents(args["env"], env_args, self.envs)
        self.agent_names = ["player_" + str(r) for r in range(self.n_agents)]

        self.iter = 0
        self.total_iter = self.algo_args["train"]["n_iter"]
        self.population_size = self.algo_args["algo"]["pop_size"]
        self.total_count = 0
        self.count = [0] * self.population_size

        self.model_config = Dotdict(self.algo_args["model"])
        self.model_config["state_shape"] = self.state_shape
        self.model_config["obs_space"] = self.obs_space[0]
        self.model_config["act_space"] = self.action_space[0]
        self.model_config["pop_size"] = self.population_size
        self.model_config["use_gpu"] = self.algo_args["device"]["cuda"]
        self.model_config["training_device"] = self.device
        self.agent_population = [
            ALGO_REGISTRY[args["algo"]](self.model_config)
            for _ in range(self.population_size)
        ]

        self.num_xp_pair_sample = min(
            algo_args["train"]["num_xp_pair_sample"], self.population_size - 1
        )
        self.sp_pairs = [(i, i) for i in range(self.population_size)]
        self.xp_pairs = [
            (i, -1)
            for i, _ in product(
                range(self.population_size), range(self.num_xp_pair_sample)
            )
        ]

        self.writer = LogWriter(args, algo_args, env_args)
        self.run_dir, self.log_dir, self.save_dir = self.writer.get_dirs()

        self.logger = LIPOLogger(
            args, algo_args, env_args, self.n_agents, self.writer, self.run_dir
        )
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )
        self.profiler = Profiler(interval=self.algo_args["logger"]["profiler_interval"])
        self.profiler.start()

    def run(self):
        print("run")
        self.iter = 0
        self.logger.init(self.total_iter)
        while self.iter < self.total_iter:
            self.logger.episode_init(self.iter, self.count)
            self.train()  # collect data and update the agents
            if ((self.iter + 1) % self.algo_args["eval"]["eval_interval"] == 0) or (
                (self.iter + 1) == self.total_iter
            ):
                self.evaluate()
            if (self.iter + 1) == self.total_iter or ((self.iter + 1) % 50 == 0):
                self.save()
            self.iter += 1

    def train(self):
        self.sample_xp_pairs()
        (
            sp_rollouts,
            away_rollouts,
            home_rollouts,
            sp_metrics,
            xp_metrics,
        ) = self.collect(train_mode=True)
        pg_mask, value_mask = self.get_mask(home_rollouts, away_rollouts)
        # train!!!
        for i in range(self.population_size):
            print(i, end=" ")
            sys.stdout.flush()
            # put only view of agent_i to agent_i.train()
            # select correct corresponding players for agent_i first before sending them to train()
            self.agent_population[i].train(
                sp_rollout=sp_rollouts[i],
                away_rollouts=away_rollouts[i],
                home_rollouts=home_rollouts[i],
                pg_mask=pg_mask[i],
                value_mask=value_mask[i],
            )
        print("train done")
        # handle metrics
        avg_metrics = dict()
        avg_metrics["sp"] = get_avg_metrics(sp_metrics)
        avg_metrics["xp"] = get_avg_metrics(xp_metrics)
        self.logger.episode_log(self.count, self.total_count, avg_metrics)

    def collect(self, train_mode):
        result = []
        n_workers = self.n_rollout_threads["train"]
        n_xp_workers = self.n_rollout_threads["xp"]
        if not train_mode:
            n_workers = self.n_rollout_threads["eval"]
            n_xp_workers = self.n_rollout_threads["eval"]

        def get_n(sp, train_mode):
            if not train_mode:
                n = self.algo_args["eval"]["n_eval_ep"]
                if self.algo_args["train"]["sample_step"]:
                    n = self.algo_args["eval"]["n_eval_ts"]
                return n
            n = (
                self.algo_args["train"]["n_sp_episodes"]
                if sp
                else self.algo_args["train"]["n_xp_episodes"]
            )
            if self.algo_args["train"]["sample_step"]:
                n = (
                    self.algo_args["train"]["n_sp_ts"]
                    if sp
                    else self.algo_args["train"]["n_xp_ts"]
                )
            return n

        def update_count(count_list, pairs, n, train_mode):
            if train_mode:
                for pair in pairs:
                    for i in pair:
                        self.count[i] += n

        # rollout trajs
        sp_rollouts = []
        sp_metric_buffer = []
        n = get_n(sp=True, train_mode=train_mode)
        update_count(self.count, self.sp_pairs, n, train_mode)
        self.total_count += n * len(self.sp_pairs)
        sp_result, sp_metrics = self.sp_rollout(n, train_mode)
        sp_rollouts.extend(sp_result)
        sp_metric_buffer.extend(sp_metrics)
        print("sp done")
        # away game -> traj that played out using param_i for one agent and param_j for the rest
        # home game -> traj that played out using param_i for all agents and param_j for one agent
        xp_home_buffer = []
        xp_away_buffer = []
        xp_metric_buffer = []
        print("xp", end=" ")
        for k in range(0, self.n_xp_pairs, n_xp_workers):
            print("#", end="")
            sys.stdout.flush()
            start = k
            end = k + n_xp_workers
            pairs = self.xp_pairs[start:end]
            n = get_n(sp=False, train_mode=train_mode)
            update_count(self.count, pairs, n, train_mode)
            self.total_count += n * len(pairs)
            home_result, away_result, xp_metrics = self.xp_rollout(pairs, n, train_mode)
            xp_home_buffer.extend(home_result)
            xp_away_buffer.extend(away_result)
            xp_metric_buffer.extend(xp_metrics)
        print(" done")
        # build home/away rollouts
        home_rollouts = [
            [None] * self.population_size for _ in range(self.population_size)
        ]
        away_rollouts = [
            [None] * self.population_size for _ in range(self.population_size)
        ]

        for i, pair in enumerate(self.xp_pairs):
            home_rollouts[pair[0]][pair[1]] = xp_home_buffer[i][0]
            away_rollouts[pair[1]][pair[0]] = xp_away_buffer[i][0]

        return (
            sp_rollouts,
            away_rollouts,
            home_rollouts,
            sp_metric_buffer,
            xp_metric_buffer,
        )

    def sp_rollout(self, n, train_mode):
        env = self.envs if train_mode else self.eval_envs
        n_workers = self.n_rollout_threads["train"]
        if not train_mode:
            n_workers = self.n_rollout_threads["eval"]
        buffer = [[] for i in range(n_workers)]
        infos = [[] for i in range(n_workers)]

        for agent in self.agent_population:
            agent.value_net = agent.sp_critic

        obs, state, avail_actions = env.reset()
        z = np.array([self.sample_z().copy() for i in range(n_workers)])
        rewards = np.zeros([n_workers, self.n_agents, 1], dtype=np.float32)
        dones = np.zeros([n_workers, self.n_agents], dtype=np.int)
        for _ in range(n):
            inps = self.pack_arrdict(
                n_workers, True, obs=obs, state=state, z=z, reward=rewards, done=dones
            )

            pair_decisions = [
                self.agent_population[i].act(inps[i]) for i in range(n_workers)
            ]
            decisions = []
            for i in range(n_workers):
                single_decision = Arrdict()
                for j, agent in enumerate(self.agent_names):
                    single_decision[agent] = pair_decisions[i][j]
                decisions.append(single_decision)

            actions = [
                [[pair_decisions[i][j].action] for j in range(self.n_agents)]
                for i in range(n_workers)
            ]
            obs, state, rewards, dones, info, available_actions = env.step(actions)

            old_obs = obs.copy()
            old_state = state.copy()
            for i, d in enumerate(dones):
                if d[0]:
                    z[i] = self.sample_z().copy()
                    old_obs[i] = info[i][0]["original_obs"]
                    old_state[i] = info[i][0]["original_state"]
            outcomes = self.pack_arrdict(
                n_workers,
                False,
                obs=old_obs,
                state=old_state,
                reward=rewards,
                done=dones,
            )

            for i in range(n_workers):
                transition = Arrdict(
                    inp=inps[i], decision=decisions[i], outcome=outcomes[i]
                )
                buffer[i].append(transition)

        trajs = []
        metrics = []
        for i in range(n_workers):
            trajs.append(arrdict.stack(buffer[i]))
            metric = Dotdict({})
            traj_info = get_traj_info(trajs[i])
            metric.update(traj_info)
            metrics.append(metric)
        return trajs, metrics

    def xp_rollout(self, pairs, n, train_mode):
        env = self.xp_envs if train_mode else self.eval_envs
        n_workers = self.n_rollout_threads["xp"]
        if not train_mode:
            n_workers = self.n_rollout_threads["eval"]
        home_trajs = [[] for i in range(n_workers)]
        away_trajs = [[] for i in range(n_workers)]
        metrics = [[] for i in range(n_workers)]

        for away_pos in range(self.n_agents):
            buffer = [[] for i in range(n_workers)]
            obs, state, avail_actions = env.reset()
            z = np.array([self.sample_z().copy() for i in range(n_workers)])
            rewards = np.zeros([n_workers, self.n_agents, 1], dtype=np.float32)
            dones = np.zeros([n_workers, self.n_agents], dtype=np.int)
            for _ in range(n // self.n_agents):
                inps = self.pack_arrdict(
                    n_workers,
                    True,
                    obs=obs,
                    state=state,
                    z=z,
                    reward=rewards,
                    done=dones,
                )

                pair_decisions = []
                for i in range(n_workers):
                    home_id, away_id = pairs[i]
                    self.agent_population[home_id].value_net = self.agent_population[
                        home_id
                    ].xp_critics[away_id]
                    self.agent_population[away_id].value_net = self.agent_population[
                        home_id
                    ].xp_critics[home_id]

                    home_decision = self.agent_population[home_id].act(inps[i])
                    away_decision = self.agent_population[away_id].act(inps[i])
                    pair_decision = []
                    for pos in range(self.n_agents):
                        if pos != away_pos:
                            pair_decision.append(home_decision[pos])
                        else:
                            pair_decision.append(away_decision[pos])
                    pair_decisions.append(pair_decision.copy())
                decisions = []
                for i in range(n_workers):
                    single_decision = Arrdict()
                    for j, agent in enumerate(self.agent_names):
                        single_decision[agent] = pair_decisions[i][j]
                    decisions.append(single_decision)

                actions = [
                    [[pair_decisions[i][j].action] for j in range(self.n_agents)]
                    for i in range(n_workers)
                ]
                obs, state, rewards, dones, info, available_actions = env.step(actions)

                old_obs = obs.copy()
                old_state = state.copy()
                for i, d in enumerate(dones):
                    if d[0]:
                        z[i] = self.sample_z().copy()
                        old_obs[i] = info[i][0]["original_obs"]
                        old_state[i] = info[i][0]["original_state"]
                outcomes = self.pack_arrdict(
                    n_workers,
                    False,
                    obs=old_obs,
                    state=old_state,
                    reward=rewards,
                    done=dones,
                )

                for i in range(n_workers):
                    transition = Arrdict(
                        inp=inps[i], decision=decisions[i], outcome=outcomes[i]
                    )
                    buffer[i].append(transition)

            trajs = []
            for i in range(n_workers):
                trajs.append(arrdict.stack(buffer[i]))
                metric = Dotdict({})
                traj_info = get_traj_info(trajs[i])
                metric.update(traj_info)
                metrics[i].append(metric)

            for i in range(n_workers):
                for pos, name in enumerate(self.agent_names):
                    if pos == away_pos:
                        if len(away_trajs[i]) == 0:
                            away_trajs[i].append(getattr(trajs[i], name))
                        else:
                            away_trajs[i][0] = arrdict.cat(
                                [away_trajs[i][0], getattr(trajs[i], name)], axis=0
                            )
                    else:
                        if len(home_trajs[i]) == 0:
                            home_trajs[i].append(getattr(trajs[i], name))
                        else:
                            home_trajs[i][0] = arrdict.cat(
                                [home_trajs[i][0], getattr(trajs[i], name)], axis=0
                            )

        full_metrics = []
        for i in range(n_workers):
            full_metrics.append(merge_dict(metrics[i]))
        return home_trajs, away_trajs, full_metrics

    def evaluate(self):
        self.sample_xp_pairs()
        (
            sp_rollouts,
            away_rollouts,
            home_rollouts,
            sp_metrics,
            xp_metrics,
        ) = self.collect(train_mode=False)
        # handle metrics
        avg_metrics = dict()
        avg_metrics["sp"] = get_avg_metrics(sp_metrics)
        avg_metrics["xp"] = get_avg_metrics(xp_metrics)
        self.logger.eval_log(avg_metrics)

    def save(self):
        print("save to:" + str(self.save_dir))
        if self.args["env"] == "overcooked":
            torch.save(self.agent_population, str(self.save_dir) + "/overcooked_population.pt")
        elif self.args["env"] == "gridworld":
            torch.save(self.agent_population, str(self.save_dir) + "/movebox_population.pt")

    def generate_payoff_matrix(self, xp_metrics):
        payoff_matrix = np.ma.zeros([self.population_size, self.population_size])
        payoff_matrix.mask = True
        for k, (i, j) in enumerate(self.xp_pairs):
            payoff_matrix[i, j] = np.round(
                np.mean(list(xp_metrics[k].avg_ret.values())), 2
            )
        return payoff_matrix

    def sample_xp_pairs(self):
        self.xp_pairs = []
        inv_eye = np.ones([self.population_size, self.population_size]) - np.eye(
            self.population_size
        )
        row, col = np.where(inv_eye > 0)
        self.xp_pairs = list(zip(row, col))

    @property
    def n_pairs(self):
        return len(self.pairs)

    @property
    def n_xp_pairs(self):
        return len(self.xp_pairs)

    @property
    def pairs(self):
        return self.sp_pairs + self.xp_pairs

    def sample_z(self):
        shared_z = self.algo_args["train"]["shared_z"]
        z_dim = self.algo_args["train"]["z_dim"]
        z_discrete = self.algo_args["train"]["z_discrete"]

        if isinstance(z_dim, int):
            if z_discrete:
                z = random.choices(np.eye(z_dim, dtype=np.float32), k=self.n_agents)
            else:
                z = np.random.uniform(
                    *self.z_range, size=(self.n_agents, self.z_dim)
                ).astype(np.float32)
            if shared_z:
                # use only one value of z
                z = np.tile(z[0], (len(z), 1))
        return np.array(z)

    def pack_arrdict(
        self, n_workers, have_data, obs=[], state=[], z=[], reward=[], done=[]
    ):
        data = []
        if have_data:
            data = [Arrdict(data=Arrdict()) for i in range(n_workers)]
        else:
            data = [Arrdict() for i in range(n_workers)]
        for w in range(n_workers):
            for i, p in enumerate(self.agent_names):
                if have_data:
                    data[w]["data"][p] = Arrdict()
                else:
                    data[w][p] = Arrdict()
                if len(obs) != 0:
                    if have_data:
                        data[w]["data"][p]["obs"] = obs[w][i].copy()
                    else:
                        data[w][p]["obs"] = obs[w][i].copy()
                if len(state) != 0:
                    if have_data:
                        data[w]["data"][p]["state"] = state[w][i].copy()
                    else:
                        data[w][p]["state"] = state[w][i].copy()
                if len(z) != 0:
                    if have_data:
                        data[w]["data"][p]["z"] = z[w][i].copy()
                    else:
                        data[w][p]["z"] = z[w][i].copy()
                if len(reward) != 0:
                    if have_data:
                        data[w]["data"][p]["reward"] = reward[w][i][0]
                    else:
                        data[w][p]["reward"] = reward[w][i][0]
                if len(done) != 0:
                    if have_data:
                        data[w]["data"][p]["done"] = done[w][i]
                    else:
                        data[w][p]["done"] = done[w][i]
        return data

    def max_mask(self, home_rollouts, away_rollouts):
        # remove data from non-max average return pair inplace
        assert len(home_rollouts) == len(away_rollouts)
        max_mask = np.zeros([len(away_rollouts), len(away_rollouts)], dtype=bool)

        for i in range(len(away_rollouts)):
            ret = np.ma.zeros(len(home_rollouts))
            ret.mask = True
            for rollouts in [home_rollouts, away_rollouts]:
                for j, r in enumerate(rollouts[i]):
                    if r is not None:
                        ret.mask[j] = False
                        ret[j] += r.outcome.reward.mean()

            idx = np.argmax(ret)
            max_mask[i, idx] = True
        return max_mask

    def get_mask(self, home_rollouts, away_rollouts):
        value_mask = np.ones([len(home_rollouts), len(home_rollouts)], dtype=np.float32)
        pg_mask = np.ones(
            [len(home_rollouts), len(home_rollouts)], dtype=np.float32
        ) / len(home_rollouts)
        # use only rollouts with highest return
        if self.algo_args["train"]["pg_xp_max_only"]:
            pg_mask = self.max_mask(home_rollouts, away_rollouts)
        if self.algo_args["train"]["value_xp_max_only"]:
            value_mask = self.max_mask(home_rollouts, away_rollouts)
        return pg_mask, value_mask

    def close(self):
        print("close")
        self.logger.close()
        self.envs.close()
        self.profiler.stop()
        with open(os.path.join(self.log_dir, "profile.html"), "w") as f:
            f.write(self.profiler.output_html())
