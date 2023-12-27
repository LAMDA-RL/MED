from harl.common.base_logger import BaseLogger


class MatrixGameLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["map_name"]
