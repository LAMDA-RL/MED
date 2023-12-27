from absl import flags

from harl.envs.matrix_game.matrix_game_logger import MatrixGameLogger
from harl.envs.gridworld.gridworld_logger import GridWorldLogger
from harl.envs.overcooked.overcooked_logger import OvercookedLogger

FLAGS = flags.FLAGS

LOGGER_REGISTRY = {
    "matrix_game": MatrixGameLogger,
    "gridworld": GridWorldLogger,
    "overcooked": OvercookedLogger,
}
