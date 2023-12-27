"""Runner registry."""
from harl.runners.generalist_runners.med_runner import MEDRunner
from harl.runners.population_runners.lipo_runner import LIPORunner

RUNNER_REGISTRY = {
    "med": MEDRunner,
    "lipo": LIPORunner,
}
