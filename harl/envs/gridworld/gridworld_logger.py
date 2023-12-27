from harl.common.base_logger import BaseLogger


class GridWorldLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["task"]
