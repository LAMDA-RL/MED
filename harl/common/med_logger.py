import os
import time

import numpy as np


class MEDLogger:
    """Base logger class.
    Used for logging information in the MED training pipeline.
    """

    def __init__(self, args, algo_args, env_args, num_agents, writer, run_dir):
        """Initialize the logger."""
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = self.get_task_name()
        self.num_agents = num_agents
        self.writer = writer
        self.run_dir = run_dir
        self.log_file = open(
            os.path.join(run_dir, "progress.txt"), "w", encoding="utf-8"
        )

    def get_task_name(self):
        """Get the task name."""
        if self.args["env"] == "gridworld":
            return self.env_args["task"]
        else:
            return self.env_args["map_name"]

    def init(self, t_max):
        """Initialize the logger."""
        self.start = time.time()
        self.t_max = t_max
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.train_episode_lengths = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        self.done_episodes_lengths = []

    def episode_init(self, episode, t_env):
        """Initialize the logger for each episode."""
        self.episode = episode
        self.t_env = t_env

    def per_step(self, data: dict, envs_not_done) -> None:
        """Process data per step."""
        dones_env = data["done"]
        reward_env = data["reward"][:, 0].copy()
        for i, e in enumerate(envs_not_done):
            self.train_episode_rewards[e] += reward_env[i]
            self.train_episode_lengths[e] += 1
            if dones_env[i]:
                self.done_episodes_rewards.append(self.train_episode_rewards[e])
                self.done_episodes_lengths.append(self.train_episode_lengths[e])
                self.train_episode_rewards[e] = 0
                self.train_episode_lengths[e] = 0

    def episode_log(self, t_env, loss=0, rewards=-999, lengths=-999):
        """Log information for each episode."""
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} Epoch {}, total num timesteps {}/{}, time {}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                t_env,
                self.t_max,
                (self.end - self.start) // 60,
                int(t_env / (self.end - self.start)),
            )
        )
        print("Training loss:", loss)
        self.writer.add_scalar(
            "loss",
            loss,
            t_env,
        )
        if len(self.done_episodes_rewards) > 0 or rewards != -999:
            if rewards != -999:
                aver_episode_rewards = rewards
            else:
                aver_episode_rewards = np.mean(self.done_episodes_rewards)
                self.done_episodes_rewards = []
            print(
                "Some episodes done, average episode reward is {}.".format(
                    np.round(aver_episode_rewards, 3)
                )
            )
            self.writer.add_scalar(
                "train_episode_reward",
                aver_episode_rewards,
                t_env,
            )

        if len(self.done_episodes_lengths) > 0 or lengths != -999:
            if lengths != -999:
                aver_episode_lengths = lengths
            else:
                aver_episode_lengths = np.mean(self.done_episodes_lengths)
                self.done_episodes_lengths = []
            print(
                "Some episodes done, average episode length is {}.".format(
                    np.round(aver_episode_lengths, 3)
                )
            )
            self.writer.add_scalar(
                "train_episode_length",
                aver_episode_lengths,
                t_env,
            )

    def eval_log(self, eval_reward, n, t_env, accuracy_lst=None):
        """Log evaluation information."""
        avg_eval_reward = [np.round(reward / n, 3) for reward in eval_reward]
        for i in range(self.algo_args["algo"]["n_episodes"]):
            self.writer.add_scalar(
                "eval_reward_episode_" + str(i),
                avg_eval_reward[i],
                t_env,
            )
        print("Evaluation average episode reward is {}.".format(avg_eval_reward))
        if accuracy_lst != None:
            avg_accuracy = [np.round(accu / n, 3) for accu in accuracy_lst]
            for i in range(self.algo_args["algo"]["n_episodes"]):
                self.writer.add_scalar(
                    "eval_accuracy_episode_" + str(i),
                    avg_accuracy[i],
                    t_env,
                )
            print("Evaluation average episode accuracy is {}.".format(avg_accuracy))
        print("\n", end="")
        self.log_file.write(",".join(map(str, [t_env, avg_eval_reward])) + "\n")
        self.log_file.flush()

    def loss_log(self, log_values, t_env):
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} Epoch {}, total num timesteps {}/{}, time {}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                t_env,
                self.t_max,
                (self.end - self.start) // 60,
                int(t_env / (self.end - self.start)),
            )
        )
        for k, v in log_values.items():
            self.writer.add_scalar(
                k,
                v,
                t_env,
            )
        print(
            "Some episodes done, policy loss {}, qf1 loss {}, qf2 loss {}, encoder loss {}, alpha_loss {}.".format(
                np.round(log_values["policy_loss"], 3),
                np.round(log_values["qf1_loss"], 3),
                np.round(log_values["qf2_loss"], 3),
                np.round(log_values["encoder_loss"], 3),
                np.round(log_values["alpha_loss"], 3),
            )
        )

    def close(self):
        """Close the logger."""
        self.log_file.close()
