from harl.common.base_logger import BaseLogger


class OvercookedLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["map_name"]
