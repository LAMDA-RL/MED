from types import SimpleNamespace as SN

import numpy as np
import torch as th


class SingleTrajectory:
    def __init__(self, scheme, max_seq_length, vocab, game="matrix_game", device="cpu"):
        self.data = SN()
        self.data.transition_data = {}
        self.data.episode_data = {}
        self.scheme = scheme.copy()
        self.seq_length = 0
        self.max_seq_length = max_seq_length
        self.vocab = vocab
        self.game = game
        self.device = device
        self._setup_data(self.scheme)

    def _setup_data(self, scheme):
        # 1. reserved key
        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({"filled": {"vshape": (1,), "dtype": th.long}})

        # 2. add preprocess keys to scheme
        tmp_scheme = {}  # store preprocess
        for field_key, field_info in scheme.items():
            if "preprocess" in field_info:
                new_k = field_info["preprocess"][0]
                transforms = field_info["preprocess"][1]
                vshape = field_info["vshape"]
                dtype = field_info["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                tmp_scheme[new_k] = {"vshape": vshape, "dtype": dtype}

                if "group" in field_info:
                    tmp_scheme[new_k]["group"] = field_info["group"]
                if "episode_const" in field_info:
                    tmp_scheme[new_k]["episode_const"] = field_key["episode_const"]
        scheme.update(tmp_scheme)

        # 3. process all the keys
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, f"Scheme must define vshape for {field_key}"
            vshape = field_info["vshape"]
            if isinstance(vshape, int):
                vshape = (vshape,)
            dtype = field_info.get("dtype", th.float32)

            # group keys should repeat for each agent
            shape = (field_info["group"], *vshape) if "group" in field_info else vshape
            # episode_const not change within an episode
            episode_const = field_info.get("episode_const", False)
            if episode_const:
                self.data.episode_data[field_key] = th.zeros(
                    (shape), dtype=dtype, device=self.device
                )
            else:
                self.data.transition_data[field_key] = th.zeros(
                    (self.max_seq_length, *shape), dtype=dtype, device=self.device
                )

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def print_info(self):
        for k, v in self.data.transition_data.items():
            print(k, v.shape)

    def return_whole_trajectory(self, player):
        obs = self.data.transition_data["obs"].clone()[:, player]
        actions = self.data.transition_data["actions"].clone()[:, player]
        reward = self.data.transition_data["reward"].clone()
        if self.game == "matrix_game":
            obs = obs.int()
            reward = reward.int()
        return obs, actions, reward

    def return_whole_trajectory_with_target(self, player):
        obs = self.data.transition_data["obs"].clone()[:, player]
        actions = self.data.transition_data["actions"].clone()[:, player]
        reward = self.data.transition_data["reward"].clone()
        if self.game == "matrix_game":
            obs = obs.int()
            reward = reward.int()

        joined_dim = 3
        mask = th.zeros(self.max_seq_length * joined_dim)
        mask[0 : self.seq_length * joined_dim : joined_dim] = 1
        target_actions = self.data.transition_data["target_actions"].clone()[:, player]
        target_pad = th.zeros([reward.size(0), reward.size(1) + 1])
        target = th.cat([target_actions, target_pad], axis=1)
        return obs, actions, reward, mask, target


class TrajectoryBuffer:
    def __init__(
        self,
        scheme,
        buffer_size,
        buffer_index,
        max_seq_length,
        vocab,
        game="matrix_game",
        device="cpu",
    ):
        self.buffer_size = buffer_size
        self.buffer_index = buffer_index
        self.trajectories_in_buffer = 0
        self.scheme = scheme
        self.max_seq_length = max_seq_length
        self.buffer = [
            SingleTrajectory(scheme, max_seq_length, vocab, game=game, device=device)
            for i in range(self.buffer_size)
        ]
        self.vocab = vocab
        self.game = game
        self.device = device
        # self.buffer[0].print_info()
        self.trajectories_length = []

    def update(self, data, bs, ts, mark_filled=True):
        index_lst = [i + self.trajectories_in_buffer for i in bs]
        for index, i in enumerate(index_lst):
            t = ts[i - self.trajectories_in_buffer]
            for k, v in data.items():
                if k in self.buffer[0].data.transition_data:
                    target = self.buffer[i].data.transition_data
                    if mark_filled:
                        self.buffer[i].data.transition_data["filled"][t] = 1
                        mark_filled = False
                elif k in self.buffer[0].data.episode_data:
                    target = self.buffer[i].data.episode_data
                else:
                    raise KeyError(f"{k} not found in transition or episode data")

                dtype = self.buffer[i].scheme[k].get("dtype", th.float32)
                vi = v[index]
                if not isinstance(vi, th.Tensor):
                    if isinstance(vi, list):
                        vi = np.array(vi)
                    vi = th.tensor(vi, dtype=dtype, device=self.device)
                else:
                    vi = vi.clone().detach()
                target[k][t] = vi.view_as(target[k][t])

                preprocess = self.buffer[i].scheme[k].get("preprocess", None)
                if preprocess:
                    new_k = preprocess[0]
                    vi_pro = target[k][t]
                    for transform in preprocess[1]:
                        vi_pro = transform.transform(vi_pro)
                    target[new_k][t] = vi_pro.view_as(target[new_k][t])

    def update_episode_finish_info(self, length):
        self.trajectories_length.append(length.copy())

    def update_finish_info(self, length):
        traj_num = len(length)
        for i in range(traj_num):
            self.buffer[i + self.trajectories_in_buffer].seq_length = length[i]
        self.trajectories_in_buffer += traj_num

    def sample(self, batch_size, player):
        o_lst = []
        a_lst = []
        r_lst = []
        target_batch = []
        mask_batch = []
        idx = np.random.randint(self.trajectories_in_buffer, size=batch_size)
        for i in idx:
            obs, action, reward, mask, target = self.buffer[
                i
            ].return_whole_trajectory_with_target(player)
            o_lst.append(obs)
            a_lst.append(action)
            r_lst.append(reward)
            mask_batch.append(mask)
            target_batch.append(target)
        return (
            th.stack(o_lst),
            th.nn.functional.one_hot(th.stack(a_lst).to(th.int64), self.vocab)
            .float()
            .squeeze(2),
            th.stack(r_lst),
            th.stack(mask_batch),
            th.stack(target_batch),
        )

    def get_transformer_input(self, envs, player, t):
        o_lst = []
        a_lst = []
        r_lst = []
        t_max = max(t)
        for i in envs:
            obs, action, reward = self.buffer[
                i + self.trajectories_in_buffer
            ].return_whole_trajectory(player)
            o_lst.append(obs[: t_max + 1])
            a_lst.append(action[: t_max + 1])
            r_lst.append(reward[: t_max + 1])
        return (
            th.stack(o_lst),
            th.nn.functional.one_hot(th.stack(a_lst).to(th.int64), self.vocab)
            .float()
            .squeeze(2),
            th.stack(r_lst),
        )

    def get_expert_input(self, n_rollout_threads, envs, players, t):
        obs = []
        avail_actions = []
        for i in range(n_rollout_threads):
            o = self.buffer[i + self.trajectories_in_buffer].data.transition_data[
                "obs"
            ][t[i]]
            aa = self.buffer[i + self.trajectories_in_buffer].data.transition_data[
                "avail_actions"
            ][t[i]]
            obs.append(o)
            avail_actions.append(aa)
        return np.stack(obs, axis=0), np.stack(avail_actions, axis=0)

    def get_eval_reward(self):
        test_reward = [0 for i in range(len(self.trajectories_length))]
        t_lst = []
        t_lst.append([0] * self.trajectories_in_buffer)
        for i in self.trajectories_length:
            t_lst.append(i)
        for i in range(self.trajectories_in_buffer):
            for j in range(len(test_reward)):
                test_reward[j] += sum(
                    self.buffer[i].data.transition_data["reward"][
                        t_lst[j][i] : t_lst[j + 1][i]
                    ]
                ).item()
        return test_reward

    def empty(self):
        self.trajectories_in_buffer = 0
        self.trajectories_length = []
        self.buffer = [
            SingleTrajectory(
                self.scheme,
                self.max_seq_length,
                self.vocab,
                game=self.game,
                device=self.device,
            )
            for i in range(self.buffer_size)
        ]

    def refresh(self, batch_size):
        new_buffer = self.buffer[
            self.trajectories_in_buffer - batch_size : self.trajectories_in_buffer
        ].copy()
        self.trajectories_length = self.trajectories_length[
            self.trajectories_in_buffer - batch_size : self.trajectories_in_buffer
        ].copy()
        self.trajectories_in_buffer = len(new_buffer)
        self.buffer = new_buffer.copy() + [
            SingleTrajectory(
                self.scheme,
                self.max_seq_length,
                self.vocab,
                game=self.game,
                device=self.device,
            )
            for i in range(self.buffer_size - self.trajectories_in_buffer)
        ]
