import os
import time

import numpy as np


class LIPOLogger:
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

    def init(self, total_iter):
        """Initialize the logger."""
        self.start = time.time()
        self.total_iter = total_iter

    def episode_init(self, n_iter, t_count):
        """Initialize the logger for each episode."""
        self.iter = n_iter
        self.t_count = t_count

    def episode_log(self, count, total_count, metrics):
        """Log information for each episode."""
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} Epoch {}/{}, timesteps per agent {}, total timesteps {}, Time {}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.iter + 1,
                self.total_iter,
                count[0],
                total_count,
                (self.end - self.start) // 60,
                int(total_count / (self.end - self.start)),
            )
        )

        for name in ["sp", "xp"]:
            print(name, end=": ")
            for k, v in metrics[name].items():
                self.writer.add_scalar(
                    name + "_" + k,
                    np.round(np.mean(v), 3),
                    self.iter,
                )
                print(k, np.round(np.mean(v), 3), end=" ")
            print("\n", end="")
        self.log_file.write(
            ",".join(
                [
                    str(self.iter),
                    "sp:" + str(np.round(np.mean(metrics["sp"]["avg_ret"]), 3)),
                    "xp:" + str(np.round(np.mean(metrics["xp"]["avg_ret"]), 3)),
                ]
            )
            + "\n"
        )
        self.log_file.flush()

    def eval_log(self, metrics):
        """Log evaluation information."""
        print("eval_result:")
        for name in ["sp", "xp"]:
            print(name, end=": ")
            for k, v in metrics[name].items():
                self.writer.add_scalar(
                    "eval" + "_" + name + "_" + k,
                    np.round(np.mean(v), 3),
                    self.iter,
                )
                print(k, np.round(np.mean(v), 3), end=" ")
            print("\n", end="")

    def close(self):
        """Close the logger."""
        self.log_file.close()
