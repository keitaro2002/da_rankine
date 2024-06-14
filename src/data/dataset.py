import os

import numpy as np

from src.config import config


class RankineData:
    """
    Example:
    ------
    >>> from src.data.dataset import RankineData
    >>> number_of_prior = 1
    >>> data = RankineData(number_of_prior)
    """

    def __init__(self, number_of_prior):
        self.xx, self.yy = np.meshgrid(np.linspace(0, 90, 91), np.linspace(0, 90, 91))
        self.params_true = {"center_x": 45, "center_y": 45, "U": 30, "R": 12}
        self.ensemble_size = config.ensemble_size
        self.number_of_prior = number_of_prior
        self.number_of_obs = config.number_of_obs
        self.radar = np.array([24, 24])
        self.PATH = "/data10/kinuki/da_rankine"
        self.noise_flag = config.noise_flag

        self.windvalues_true = None
        self.windvalues_prior = None
        self.obs_points = None
        self.obs = None

        # 初期化
        self.initialize_data()

    def initialize_data(self):
        files = {
            "true": "windvalues_true.npz",
            "prior": f"windvalues_prior{self.number_of_prior}.npz",
            "obs_points": "obs_points.npy",
            "obs": "windvalues_obs.npz" if self.noise_flag else "windvalues_obs_no_noise.npz",
        }
        # windvalues_trueのデータ
        true_path = f'{self.PATH}/data/{files["true"]}'
        if os.path.exists(true_path):
            self.windvalues_true = self.load_data(files["true"])
        else:
            self.windvalues_true = self.wind_speed(self.xx, self.yy, self.params_true)
            self.save_data(files["true"], self.windvalues_true)

        # windvalues_priorのデータ
        prior_path = f'{self.PATH}/data/{files["prior"]}'
        if os.path.exists(prior_path):
            self.windvalues_prior = self.load_data(files["prior"])
        else:
            if not os.path.exists(self.PATH):
                os.makedirs(self.PATH)
            self.params_prior = self.make_prior_params()
            self.windvalues_prior = self.calculate_windspeeds_prior()

        # obs_pointsのデータ
        obs_points_path = f'{self.PATH}/data/{files["obs_points"]}'
        if os.path.exists(obs_points_path):
            self.obs_points = self.load_data(files["obs_points"])
        else:
            self.obs_points = self.decide_obs_points()

        # obsのデータ
        obs_path = f'{self.PATH}/data/{files["obs"]}'
        print(files["obs"])
        if os.path.exists(obs_path):
            self.obs = self.load_data(files["obs"])
        else:
            self.obs = self.make_obs()

    def save_data(self, filename, data):
        if isinstance(data, dict):
            # 辞書型の場合、np.savezを使用
            np.savez(f"{self.PATH}/data/{filename}.npz", **data)
        else:
            # NumPy型の場合、np.saveを使用
            np.save(f"{self.PATH}/data/{filename}.npy", data)

    def load_data(self, filename):
        return np.load(f"{self.PATH}/data/{filename}", allow_pickle=True)

    def wind_speed(self, xx, yy, params):
        c_x, c_y, U, R = params["center_x"], params["center_y"], params["U"], params["R"]
        dx = xx - c_x
        dy = yy - c_y
        r = np.sqrt(dx**2 + dy**2)
        u = np.where(r <= R, U * r / R, U * R / r)
        u_x = np.where(r != 0, -u * dy / r, 0)
        u_y = np.where(r != 0, u * dx / r, 0)
        # 辞書型にまとめる．
        return {"u_x": u_x, "u_y": u_y, "u": u}

    def make_prior_params(self):

        """params_prior

        type: dict型
        key: center_x, center_y, U, R

        それぞれのkeyに対して，ensemble_sizeの数だけの乱数を生成する．
        1. number_of_prior=1,2,3の時
            U, R は値を固定する．
            center_x, center_y は，それぞれの値を平均として，標準偏差がstd_devの正規分布に従う乱数を生成する．
        2. number_of_prior=4の時
            U, R はN(U_true, 3^2), N(R_true, 3^2)の正規分布に従う乱数を生成する．
            std_dev = サンプリングされたRの値
            center_x, center_y は，それぞれの値を平均として，標準偏差がstd_devの正規分布に従う乱数を生成する．

        """

        if self.number_of_prior in [1, 2, 3]:
            if self.number_of_prior == 1:
                ratio = 0.25
            elif self.number_of_prior == 2:
                ratio = 0.5
            elif self.number_of_prior == 3:
                ratio = 1.0

            std_dev = ratio * self.params_true["R"]
            params_prior = {}
            for key in self.params_true.keys():
                if key == "U" or key == "R":
                    params_prior[key] = np.repeat(self.params_true[key], self.ensemble_size)
                else:
                    params_prior[key] = np.random.normal(
                        self.params_true[key], std_dev, self.ensemble_size
                    )

        elif self.number_of_prior == 4:
            params_prior = {}
            for key in self.params_true.keys():
                if key == "U" or key == "R":
                    params_prior[key] = np.random.normal(
                        self.params_true[key], 3, self.ensemble_size
                    )
                else:
                    for i in range(self.ensemble_size):
                        params_prior[key][i] = np.random.normal(
                            self.params_true[key], self.params_true["R"][i]
                        )

        self.save_data(f"params_prior{self.number_of_prior}", params_prior)
        return params_prior

    def calculate_windspeeds_prior(self):
        windspeeds_prior = {
            "u_x": np.zeros((self.ensemble_size, 91, 91)),
            "u_y": np.zeros((self.ensemble_size, 91, 91)),
            "u": np.zeros((self.ensemble_size, 91, 91)),
        }

        for i in range(self.ensemble_size):
            params = {key: self.params_prior[key][i] for key in self.params_prior}
            wind = self.wind_speed(self.xx, self.yy, params)
            windspeeds_prior["u_x"][i] = wind["u_x"]
            windspeeds_prior["u_y"][i] = wind["u_y"]
            windspeeds_prior["u"][i] = wind["u"]

        self.save_data(f"windvalues_prior{self.number_of_prior}", windspeeds_prior)
        return windspeeds_prior

    def decide_obs_points(self):
        possible_values = list(range(91))
        possible_values.remove(24)
        obs_points = []
        while len(obs_points) < self.number_of_obs:
            value = np.random.choice(possible_values, 2).tolist()
            if value not in obs_points:
                obs_points.append(value)
        obs_points = np.array(obs_points)

        self.save_data(f"obs_points", obs_points)
        return obs_points

    def make_obs(self, noise_flag=True):
        center = np.array([self.params_true["center_x"], self.params_true["center_y"]])
        radar = self.radar
        n_obs = self.number_of_obs
        if os.path.exists(f"{self.PATH}/data/obs_points.npy"):
            obs_points = self.load_data("obs_points.npy")
        else:
            obs_points = self.decide_obs_points()
        dist_obs_radar = np.sqrt(
            (obs_points[:, 0] - radar[0]) ** 2 + (obs_points[:, 1] - radar[1]) ** 2
        )
        dist_obs_center = np.sqrt(
            (obs_points[:, 0] - center[0]) ** 2 + (obs_points[:, 1] - center[1]) ** 2
        )

        housen = (
            np.vstack((-(obs_points[:, 1] - center[1]), obs_points[:, 0] - center[0])).T
            / dist_obs_center[:, None]
        )
        condition = (center[0] - radar[0]) * (obs_points[:, 1] - radar[1]) - (
            center[1] - radar[1]
        ) * (obs_points[:, 0] - radar[0]) >= 0
        vec_obs_radar_x = (
            np.where(condition, radar[0] - obs_points[:, 0], obs_points[:, 0] - radar[0])
            / dist_obs_radar
        )
        vec_obs_radar_y = (
            np.where(condition, radar[1] - obs_points[:, 1], obs_points[:, 1] - radar[1])
            / dist_obs_radar
        )
        vec_obs_radar = np.vstack((vec_obs_radar_x, vec_obs_radar_y)).T

        cos = np.sum(housen * vec_obs_radar, axis=1)
        tru_wind = self.wind_speed(obs_points[:, 0], obs_points[:, 1], self.params_true)["u"] * cos
        values = np.where(condition, -tru_wind, tru_wind)
        if noise_flag:
            noise = np.random.normal(0, 3, n_obs)
        else:
            noise = np.zeros(n_obs)
        values_withnoise = values + noise

        uv = vec_obs_radar * np.abs(values_withnoise[:, None])
        condition = (values * values_withnoise) < 0
        condition_reshaped = condition[:, np.newaxis]
        uv_final = np.where(condition_reshaped, -uv, uv)

        obs = {"u_x": uv_final[:, 0], "u_y": uv_final[:, 1], "u": values_withnoise}
        if noise_flag:
            self.save_data(f"windvalues_obs", obs)
        else:
            self.save_data(f"windvalues_obs_no_noise", obs)
        return obs


if __name__ == "__main__":
    number_of_prior = 1
    data = RankineData(number_of_prior)
    print(data.windvalues_prior["u_x"].shape)
    print(data.obs_points.shape)
    print(data.obs["u_x"].shape)
    print(data.obs["u"].shape)
