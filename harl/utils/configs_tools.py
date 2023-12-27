"""Tools for loading and updating configs."""
import json
import os
import time
from uu import Error

import yaml


def get_defaults_yaml_args(algo, env):
    """Load config file for user-specified algo and env.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args


def update_args(unparsed_dict, *args):
    """Update loaded config with unparsed command-line arguments.
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)


def get_task_name(env, env_args):
    """Get task name."""
    if env == "matrix_game":
        task = env_args["map_name"]
    elif env == "gridworld":
        task = env_args["task"]
    elif env == "overcooked":
        task = env_args["map_name"]
    return task


def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


class LogWriter:
    """Logger class for writing log to file, supporting "tensorboardX" and "wandb" with "tensorboardX" interface."""

    def __init__(self, args, algo_args, env_args):
        # Init directory for saving results.
        config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
        env, algo, exp_name = args["env"], args["algo"], args["exp_name"]
        logger_dir, seed = algo_args["logger"]["log_dir"], algo_args["seed"]["seed"]

        task = get_task_name(env, env_args)  # name of the map, layout, etc
        hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        run_name = "-".join(["seed-{:0>5}".format(seed), hms_time])  # name of this run
        self.run_dir = os.path.join(logger_dir, env, task, algo, exp_name, run_name)

        self.log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.save_dir = os.path.join(self.run_dir, "models")
        os.makedirs(self.save_dir, exist_ok=True)

        self._save_config(config)
        self._init_loggers(algo_args, env, task, exp_name, run_name, config)

    def _save_config(self, config):
        """Save the configuration of the program."""
        config_json = convert_json(config)
        output = json.dumps(
            config_json, separators=(",", ":\t"), indent=4, sort_keys=True
        )
        with open(
            os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8"
        ) as out:
            out.write(output)

    def _init_loggers(self, algo_args, env, task, exp_name, run_name, config):
        """Initialize external loggers. TODO: make it optional"""
        self.use_tensorboard = algo_args["logger"]["use_tensorboard"]
        if self.use_tensorboard:
            from tensorboardX import SummaryWriter

            tb_path = os.path.join(self.log_dir, "tensorboard")
            os.makedirs(tb_path, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_path)

        self.use_wandb = algo_args["logger"]["use_wandb"]
        if self.use_wandb:
            import wandb

            os.environ["WANDB_MODE"] = "offline"
            self.wandb_run = wandb.init(
                project="CTC",  # project level constant
                group=".".join([env, task]),
                job_type=exp_name,
                name=run_name,
                dir=self.log_dir,  # will automatically create "wandb" subfolder
                reinit=True,
                config=config,
            )

    def get_dirs(self):
        """Get the directories of the run."""
        return self.run_dir, self.log_dir, self.save_dir

    def add_scalar(self, tag, scalar_value, global_step=None):
        if self.use_tensorboard:
            self.tb_writer.add_scalar(tag, scalar_value, global_step)
        if self.use_wandb:
            self.wandb_run.log({tag: scalar_value}, global_step)

    def export_scalars_to_json(self, path):
        if self.use_tensorboard:
            self.tb_writer.export_scalars_to_json(path)

    def close(self):
        if self.use_tensorboard:
            self.tb_writer.close()
        if self.use_wandb:
            self.wandb_run.finish()
