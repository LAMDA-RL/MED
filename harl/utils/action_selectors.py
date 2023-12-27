import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical


class OneHot:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32


class DecayThenFlatSchedule:
    def __init__(self, start, finish, time_length, decay="exp"):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (
                (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1
            )

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(-T / self.exp_scaling)))


class EpsilonGreedyActionSelector:
    def __init__(self, epsilon_start, epsilon_finish, epsilon_anneal_time):
        self.schedule = DecayThenFlatSchedule(
            epsilon_start, epsilon_finish, epsilon_anneal_time, decay="linear"
        )
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, available_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)
        pick_epsilon = 0 if test_mode else self.epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[available_actions == 0.0] = -float(
            "inf"
        )  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < pick_epsilon).long()
        random_actions = Categorical(available_actions.float()).sample().long()
        picked_actions = (
            pick_random * random_actions
            + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        )
        return picked_actions


class MultinomialActionSelector:
    def __init__(self):
        self.test_greedy = True

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = -1e7
        masked_policies = th.softmax(masked_policies, dim=-1)
        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=1)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions
