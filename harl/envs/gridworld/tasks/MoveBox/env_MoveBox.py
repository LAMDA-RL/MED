import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


class EnvMoveBox(object):
    def __init__(self, horizon, map_name, obs_type):
        self.original_map = self.load_map(map_name)
        self.original_agt1_pos = [
            np.where(self.original_map == 2)[0][0],
            np.where(self.original_map == 2)[1][0],
        ]
        self.original_agt2_pos = [
            np.where(self.original_map == 3)[0][0],
            np.where(self.original_map == 3)[1][0],
        ]
        self.original_box_pos = [
            np.where(self.original_map == 4)[0][0],
            np.where(self.original_map == 4)[1][0],
        ]

        exits = np.where(self.original_map == 5)
        self.exit_pos = [[exits[0][i], exits[1][i]] for i in range(len(exits[0]))]
        far_exits = np.where(self.original_map == 6)
        self.far_exit_pos = [
            [far_exits[0][i], far_exits[1][i]] for i in range(len(far_exits[0]))
        ]

        self.map_size = [len(self.original_map), len(self.original_map[0])]

        self.raw_occupancy = self.original_map.copy()
        self.raw_occupancy[np.where(self.raw_occupancy > 1)] = 0
        self.reset()
        # print(self.original_map)
        # print(self.original_agt1_pos, self.original_agt2_pos, self.original_box_pos)
        # print(self.exit_pos)

        self.time = 0
        self.horizon = horizon
        self.obs_type = obs_type

    def load_map(self, map_name):
        with open(
            "../harl/envs/gridworld/tasks/MoveBox/maps/" + map_name + ".txt", "r"
        ) as f:
            lines = f.readlines()
        m = []
        for line in lines:
            line_lst = []
            for letter in line:
                if letter != "\n":
                    line_lst.append(int(letter))
            m.append(line_lst)
        return np.array(m)

    def reset(self):
        self.time = 0
        self.agt1_pos = self.original_agt1_pos.copy()
        self.agt2_pos = self.original_agt2_pos.copy()
        self.box_pos = self.original_box_pos.copy()

        self.occupancy = self.raw_occupancy.copy()
        self.occupancy[self.agt1_pos[0], self.agt1_pos[1]] = 1
        self.occupancy[self.agt2_pos[0], self.agt2_pos[1]] = 1
        self.occupancy[self.box_pos[0], self.box_pos[1]] = 1

        self.is_1_catch_box = False
        self.is_2_catch_box = False
        if (
            self.agt1_pos[0] == self.box_pos[0]
            and abs(self.agt1_pos[1] - self.box_pos[1]) == 1
        ):
            self.is_1_catch_box = True
        if (
            self.agt2_pos[0] == self.box_pos[0]
            and abs(self.agt2_pos[1] - self.box_pos[1]) == 1
        ):
            self.is_2_catch_box = True

    def step(self, action_list):
        if self.is_1_catch_box == False:
            if action_list[0] == 0:  # up
                if (
                    self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1]] != 1
                ):  # if can move
                    self.agt1_pos[0] = self.agt1_pos[0] - 1
                    self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1]] = 0
                    self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            if action_list[0] == 1:  # down
                if (
                    self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1]] != 1
                ):  # if can move
                    self.agt1_pos[0] = self.agt1_pos[0] + 1
                    self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1]] = 0
                    self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            if action_list[0] == 2:  # left
                if (
                    self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] - 1] != 1
                ):  # if can move
                    self.agt1_pos[1] = self.agt1_pos[1] - 1
                    self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] + 1] = 0
                    self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            if action_list[0] == 3:  # right
                if (
                    self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] + 1] != 1
                ):  # if can move
                    self.agt1_pos[1] = self.agt1_pos[1] + 1
                    self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] - 1] = 0
                    self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1

        if self.is_2_catch_box == False:
            if action_list[1] == 0:  # up
                if (
                    self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1]] != 1
                ):  # if can move
                    self.agt2_pos[0] = self.agt2_pos[0] - 1
                    self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1]] = 0
                    self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            if action_list[1] == 1:  # down
                if (
                    self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1]] != 1
                ):  # if can move
                    self.agt2_pos[0] = self.agt2_pos[0] + 1
                    self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1]] = 0
                    self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            if action_list[1] == 2:  # left
                if (
                    self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] - 1] != 1
                ):  # if can move
                    self.agt2_pos[1] = self.agt2_pos[1] - 1
                    self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] + 1] = 0
                    self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            if action_list[1] == 3:  # right
                if (
                    self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] + 1] != 1
                ):  # if can move
                    self.agt2_pos[1] = self.agt2_pos[1] + 1
                    self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] - 1] = 0
                    self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1

        if self.is_1_catch_box and self.is_2_catch_box:
            if action_list[0] == 0 and action_list[1] == 0:  # up
                if (
                    self.occupancy[self.box_pos[0] - 1, self.box_pos[1]] == 0
                    and self.occupancy[self.box_pos[0] - 1, self.box_pos[1] - 1] == 0
                    and self.occupancy[self.box_pos[0] - 1, self.box_pos[1] + 1] == 0
                ):
                    self.box_pos[0] = self.box_pos[0] - 1
                    self.agt1_pos[0] = self.agt1_pos[0] - 1
                    self.agt2_pos[0] = self.agt2_pos[0] - 1
                    self.occupancy[self.box_pos[0] + 1, self.box_pos[1]] = 0
                    self.occupancy[self.agt1_pos[0] + 1, self.agt1_pos[1]] = 0
                    self.occupancy[self.agt2_pos[0] + 1, self.agt2_pos[1]] = 0
                    self.occupancy[self.box_pos[0], self.box_pos[1]] = 1
                    self.occupancy[self.agt1_pos[0], self.agt1_pos[1]] = 1
                    self.occupancy[self.agt2_pos[0], self.agt2_pos[1]] = 1
            if action_list[0] == 1 and action_list[1] == 1:  # down
                if (
                    self.occupancy[self.box_pos[0] + 1, self.box_pos[1]] == 0
                    and self.occupancy[self.box_pos[0] + 1, self.box_pos[1] - 1] == 0
                    and self.occupancy[self.box_pos[0] + 1, self.box_pos[1] + 1] == 0
                ):
                    self.box_pos[0] = self.box_pos[0] + 1
                    self.agt1_pos[0] = self.agt1_pos[0] + 1
                    self.agt2_pos[0] = self.agt2_pos[0] + 1
                    self.occupancy[self.box_pos[0] - 1, self.box_pos[1]] = 0
                    self.occupancy[self.agt1_pos[0] - 1, self.agt1_pos[1]] = 0
                    self.occupancy[self.agt2_pos[0] - 1, self.agt2_pos[1]] = 0
                    self.occupancy[self.box_pos[0], self.box_pos[1]] = 1
                    self.occupancy[self.agt1_pos[0], self.agt1_pos[1]] = 1
                    self.occupancy[self.agt2_pos[0], self.agt2_pos[1]] = 1
            if action_list[0] == 2 and action_list[1] == 2:  # left
                if self.occupancy[self.box_pos[0], self.box_pos[1] - 2] == 0:
                    self.box_pos[1] = self.box_pos[1] - 1
                    self.agt1_pos[1] = self.agt1_pos[1] - 1
                    self.agt2_pos[1] = self.agt2_pos[1] - 1
                    self.occupancy[self.box_pos[0], self.box_pos[1] - 1] = 1
                    self.occupancy[self.box_pos[0], self.box_pos[1] + 2] = 0
            if action_list[0] == 3 and action_list[1] == 3:  # right
                if self.occupancy[self.box_pos[0], self.box_pos[1] + 2] == 0:
                    self.box_pos[1] = self.box_pos[1] + 1
                    self.agt1_pos[1] = self.agt1_pos[1] + 1
                    self.agt2_pos[1] = self.agt2_pos[1] + 1
                    self.occupancy[self.box_pos[0], self.box_pos[1] + 1] = 1
                    self.occupancy[self.box_pos[0], self.box_pos[1] - 2] = 0

        if (
            self.agt1_pos[0] == self.box_pos[0]
            and abs(self.agt1_pos[1] - self.box_pos[1]) == 1
        ):
            self.is_1_catch_box = True

        if (
            self.agt2_pos[0] == self.box_pos[0]
            and abs(self.agt2_pos[1] - self.box_pos[1]) == 1
        ):
            self.is_2_catch_box = True

        self.time += 1
        done = False
        reward = 0
        if self.box_pos in self.exit_pos:
            reward = 1
            done = True
            self.reset()
        elif self.box_pos in self.far_exit_pos:
            reward = 2
            done = True
            self.reset()
        elif self.time == self.horizon:
            done = True
            self.reset()
        return reward, done

    def get_global_obs(self, obs_type="dense"):
        if obs_type == "dense":
            obs = self.raw_occupancy.copy()
            obs[self.agt1_pos[0]][self.agt1_pos[1]] = 2
            obs[self.agt2_pos[0]][self.agt2_pos[1]] = 3
            obs[self.box_pos[0]][self.box_pos[1]] = 4
            return obs.astype(np.float).reshape(-1)
        else:
            obs = np.ones((self.map_size[0], self.map_size[1], 3))
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    if self.raw_occupancy[i, j] == 1:
                        obs[i, j, 0] = 0.0
                        obs[i, j, 1] = 0.0
                        obs[i, j, 2] = 0.0
            obs[self.agt1_pos[0], self.agt1_pos[1], 0] = 1
            obs[self.agt1_pos[0], self.agt1_pos[1], 1] = 0
            obs[self.agt1_pos[0], self.agt1_pos[1], 2] = 0

            obs[self.agt2_pos[0], self.agt2_pos[1], 0] = 0
            obs[self.agt2_pos[0], self.agt2_pos[1], 1] = 0
            obs[self.agt2_pos[0], self.agt2_pos[1], 2] = 1

            obs[self.box_pos[0], self.box_pos[1], 0] = 0
            obs[self.box_pos[0], self.box_pos[1], 1] = 1
            obs[self.box_pos[0], self.box_pos[1], 2] = 0
            if obs_type == "flatten_img":
                obs = obs.reshape(-1)
            return obs

    def get_agt1_obs(self, obs_type="dense"):
        if obs_type == "dense":
            obs = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    obs[i][j] = self.raw_occupancy[self.agt1_pos[0] - 1 + i][
                        self.agt1_pos[1] - 1 + j
                    ]
            obs[1][1] = 2
            d_x = self.agt2_pos[0] - self.agt1_pos[0]
            d_y = self.agt2_pos[1] - self.agt1_pos[1]
            if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                obs[1 + d_x, 1 + d_y] = 3
            d_x = self.box_pos[0] - self.agt1_pos[0]
            d_y = self.box_pos[1] - self.agt1_pos[1]
            if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                obs[1 + d_x, 1 + d_y] = 4
            return obs.reshape(-1)
        else:
            obs = np.zeros((3, 3, 3))
            for i in range(3):
                for j in range(3):
                    if (
                        self.raw_occupancy[self.agt1_pos[0] - 1 + i][
                            self.agt1_pos[1] - 1 + j
                        ]
                        == 0
                    ):
                        obs[i, j, 0] = 1.0
                        obs[i, j, 1] = 1.0
                        obs[i, j, 2] = 1.0
                    d_x = self.agt2_pos[0] - self.agt1_pos[0]
                    d_y = self.agt2_pos[1] - self.agt1_pos[1]
                    if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                        obs[1 + d_x, 1 + d_y, 0] = 0.0
                        obs[1 + d_x, 1 + d_y, 1] = 0.0
                        obs[1 + d_x, 1 + d_y, 2] = 1.0
                    d_x = self.box_pos[0] - self.agt1_pos[0]
                    d_y = self.box_pos[1] - self.agt1_pos[1]
                    if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                        obs[1 + d_x, 1 + d_y, 0] = 0.0
                        obs[1 + d_x, 1 + d_y, 1] = 1.0
                        obs[1 + d_x, 1 + d_y, 2] = 0.0
            obs[1, 1, 0] = 1.0
            obs[1, 1, 1] = 0.0
            obs[1, 1, 2] = 0.0
            if obs_type == "flatten_img":
                obs = obs.reshape(-1)
            return obs

    def get_agt2_obs(self, obs_type="dense"):
        if obs_type == "dense":
            obs = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    obs[i][j] = self.raw_occupancy[self.agt1_pos[0] - 1 + i][
                        self.agt1_pos[1] - 1 + j
                    ]
            obs[1][1] = 3
            d_x = self.agt1_pos[0] - self.agt2_pos[0]
            d_y = self.agt1_pos[1] - self.agt2_pos[1]
            if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                obs[1 + d_x, 1 + d_y] = 2
            d_x = self.box_pos[0] - self.agt2_pos[0]
            d_y = self.box_pos[1] - self.agt2_pos[1]
            if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                obs[1 + d_x, 1 + d_y] = 4
            return obs.reshape(-1)
        else:
            obs = np.zeros((3, 3, 3))
            for i in range(3):
                for j in range(3):
                    if (
                        self.raw_occupancy[self.agt2_pos[0] - 1 + i][
                            self.agt2_pos[1] - 1 + j
                        ]
                        == 0
                    ):
                        obs[i, j, 0] = 1.0
                        obs[i, j, 1] = 1.0
                        obs[i, j, 2] = 1.0
                    d_x = self.agt1_pos[0] - self.agt2_pos[0]
                    d_y = self.agt1_pos[1] - self.agt2_pos[1]
                    if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                        obs[1 + d_x, 1 + d_y, 0] = 1.0
                        obs[1 + d_x, 1 + d_y, 1] = 0.0
                        obs[1 + d_x, 1 + d_y, 2] = 0.0
                    d_x = self.box_pos[0] - self.agt2_pos[0]
                    d_y = self.box_pos[1] - self.agt2_pos[1]
                    if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                        obs[1 + d_x, 1 + d_y, 0] = 0.0
                        obs[1 + d_x, 1 + d_y, 1] = 1.0
                        obs[1 + d_x, 1 + d_y, 2] = 0.0
            obs[1, 1, 0] = 0.0
            obs[1, 1, 1] = 0.0
            obs[1, 1, 2] = 1.0
            if obs_type == "flatten_img":
                obs = obs.reshape(-1)
            return obs

    def get_state(self):
        state = np.zeros((1, 6))
        state[0, 0] = self.agt1_pos[0] / 15
        state[0, 1] = self.agt1_pos[1] / 15
        state[0, 2] = self.agt2_pos[0] / 15
        state[0, 3] = self.agt2_pos[1] / 15
        state[0, 4] = self.box_pos[0] / 15
        state[0, 5] = self.box_pos[1] / 15
        return state

    def get_obs(self):
        return [
            self.get_agt1_obs(obs_type=self.obs_type),
            self.get_agt2_obs(obs_type=self.obs_type),
        ]

    def plot_scene(self):
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[0, 1])
        ax1.imshow(self.get_global_obs(obs_type="img"))
        plt.xticks([])
        plt.yticks([])
        ax2.imshow(self.get_agt1_obs(obs_type="img"))
        plt.xticks([])
        plt.yticks([])
        ax3.imshow(self.get_agt2_obs(obs_type="img"))
        plt.xticks([])
        plt.yticks([])
        ax4.imshow(self.occupancy)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def render(self):
        enlarge = 20
        obs = np.ones(
            (len(self.raw_occupancy) * enlarge, len(self.raw_occupancy[0]) * enlarge, 3)
        )
        for i in range(len(self.raw_occupancy)):
            for j in range(len(self.raw_occupancy[0])):
                if self.raw_occupancy[i, j] == 1:
                    obs[
                        i * enlarge : i * enlarge + enlarge,
                        j * enlarge : j * enlarge + enlarge,
                        0,
                    ] = 0
                    obs[
                        i * enlarge : i * enlarge + enlarge,
                        j * enlarge : j * enlarge + enlarge,
                        1,
                    ] = 0
                    obs[
                        i * enlarge : i * enlarge + enlarge,
                        j * enlarge : j * enlarge + enlarge,
                        2,
                    ] = 0
        pos_lst = [self.agt1_pos, self.agt2_pos, self.box_pos]
        color_lst = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        for k in range(3):
            i, j = pos_lst[k]
            color = color_lst[k]
            obs[
                i * enlarge : i * enlarge + enlarge,
                j * enlarge : j * enlarge + enlarge,
                0,
            ] = color[0]
            obs[
                i * enlarge : i * enlarge + enlarge,
                j * enlarge : j * enlarge + enlarge,
                1,
            ] = color[1]
            obs[
                i * enlarge : i * enlarge + enlarge,
                j * enlarge : j * enlarge + enlarge,
                2,
            ] = color[2]
        return np.uint8(obs) * 255
