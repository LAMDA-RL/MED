"""Algorithm registry."""
# lipo
from harl.algorithms.actors.incompact_mappo_z import IncompatMAPPOZ
from harl.algorithms.actors.med_gpt import GPTAgent

ALGO_REGISTRY = {
    # population
    "lipo": IncompatMAPPOZ,
    # generalist
    "med": GPTAgent,
}
