import logging
import time
from contextlib import contextmanager
from functools import partial

import pandas as pd
import torch
from torch import nn

from .. import paths

log = logging.getLogger(__name__)

from . import gpu
from .reading import Reader, from_dir

# For re-export
from .writing import *
from .writing import mean, to_dir


@contextmanager
def via_dir(run_name, *args, **kwargs):
    with to_dir(run_name), from_dir(run_name, *args, **kwargs):
        yield


def normhook(name, t):
    def hook(grad):
        mean(name, grad.pow(2).sum().pow(0.5))

    t.register_hook(hook)


def total_gradient_norm(params):
    if isinstance(params, nn.Module):
        return total_gradient_norm(params.parameters())
    norms = [p.grad.data.float().pow(2).sum() for p in params if p.grad is not None]
    return torch.sum(torch.tensor(norms)).pow(0.5)


def total_norm(params):
    if isinstance(params, nn.Module):
        return total_norm(params.parameters())
    return sum([p.data.float().pow(2).sum() for p in params if p is not None]).pow(0.5)


def rel_gradient_norm(name, agent):
    mean(name, total_gradient_norm(agent), total_norm(agent))


def funcduty(name):
    def factory(f):
        def g(self, *args, **kwargs):
            start = time.time()
            result = f(self, *args, **kwargs)
            record("duty", f"duty/{name}", time.time() - start)
            return result

        return g

    return factory


def compare(run_names=[-1], prefix="", rule="60s"):
    return pd.concat(
        {paths.resolve(run): Reader(run, prefix).resample(rule) for run in run_names}, 1
    )


## TESTS


def test_from_dir():
    from .. import logging, paths, widgets

    paths.clear("test-run", "stats")
    paths.clear("test-run", "logs")

    compositor = widgets.Compositor()
    with logging.from_dir("test-run", compositor), to_dir("test-run"), from_dir(
        "test-run", compositor
    ):
        for i in range(10):
            mean("count", i)
            mean("twocount", 2 * i)
            time.sleep(0.25)
