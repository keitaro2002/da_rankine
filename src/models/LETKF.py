import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

from src.config import config
from src.data.dataset import RankineData


class LETKF:
    def __init__(self):
        self.name = "LETKF"

        self.ensemble_size = config.ensemble_size
        self.number_of_obs = config.number_of_obs
        self.data = RankineData(config.number_of_prior)
        self.noise = "" if config.noise_flag else "no_noise_"

        self.H = None
        self.L_loc = None

    def localizescale(self, sigma, distances):
        condition = 2 * np.sqrt(10 / 3) * sigma
        return np.where(distances < condition, np.exp(-(distances**2) / (2 * sigma**2)), 0)

    def make_localization_matrix(self, sigma):
        # ここも計算時間が2.2秒くらいかかる．
        xx, yy = self.data.xx, self.data.yy
        num_points = 91
        center = np.zeros((num_points, num_points, 2))
        center[:, :, 0] = xx
        center[:, :, 1] = yy
        center = center.reshape(-1, 2)

        result = np.zeros((num_points**2, num_points**2))

        for i, (center_x, center_y) in enumerate(center):
            distances = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

            gaussian = self.localizescale(sigma, distances)
            result[i, :] = gaussian.flatten()
        print(result.shape)
        self.data.save_data("localization_matrix", result)
        self.L_loc = result

    def make_observation_matrix(self):
        H = np.zeros((self.number_of_obs, 91 * 91))
        flatten_obs_points = self.flatten_obs_points()
        for i, point in enumerate(flatten_obs_points):
            H[i][int(point)] = 1
        self.H = H

    def flatten_obs_points(self):
        obs_points, _, _ = self.sort_obs()
        flatten_obs_points = np.zeros(len(obs_points), dtype=int)
        for i in range(len(obs_points)):
            flatten_obs_points[i] = obs_points[i][0] + (91 * obs_points[i][1])
        print(flatten_obs_points[:10])
        return flatten_obs_points

    def sort_obs(self):
        sorted_indices = np.lexsort((self.data.obs_points[:, 0], self.data.obs_points[:, 1]))
        sorted_obs_points = self.data.obs_points[sorted_indices]
        sorted_obs = np.zeros((self.number_of_obs, 2))
        for i in range(self.number_of_obs):
            sorted_obs[i][0], sorted_obs[i][1] = (
                self.data.obs["u_x"][sorted_indices[i]],
                self.data.obs["u_y"][sorted_indices[i]],
            )
        return sorted_obs_points, sorted_obs[:, 0], sorted_obs[:, 1]

    def calculate_R_inv(self):
        R = np.eye(91 * 91) * (3**2)
        R_inv = LA.inv(R)
        self.save_data("R_inv", R_inv)

    def preparation_for_analysis(self):
        x_u, x_v = self.data.windvalues_prior["u_x"], self.data.windvalues_prior["u_y"]
        x_u, x_v = x_u.reshape(self.ensemble_size, -1), x_v.reshape(self.ensemble_size, -1)
        xb_mean_u = np.mean(x_u, axis=0)
        xb_mean_v = np.mean(x_v, axis=0)
        dXb_u = (x_u - xb_mean_u).T
        dXb_v = (x_v - xb_mean_v).T
        dYb_u = self.H @ dXb_u
        dYb_v = self.H @ dXb_v
        R_inv = np.load(f"{self.data.PATH}/data//R_inv.npy")

        return xb_mean_u, xb_mean_v, dXb_u, dXb_v, dYb_u, dYb_v, R_inv

    def parallel_analysis(self, p_range, delta):
        N = 91 * 91
        n_obs = self.number_of_obs
        obs_points, obs_u, obs_v = self.sort_obs()
        flatten_obs_points = self.flatten_obs_points()
        H = self.H
        xb_mean_u, xb_mean_v, dXb_u, dXb_v, dYb_u, dYb_v, R_inv = self.preparation_for_analysis()

        xa_u = np.zeros((self.ensemble_size, len(p_range)))
        xa_v = np.zeros((self.ensemble_size, len(p_range)))

        for idx, p in enumerate(p_range):
            print(p)
            for j in range(N):
                R_inv[j][j] = R_inv[j][j] * self.L_loc[p][j]
            R_inv_sensitive = np.zeros((n_obs, n_obs))
            for j in range(n_obs):
                R_inv_sensitive[j][j] = R_inv[flatten_obs_points[j]][flatten_obs_points[j]]
            A_u = (self.ensemble_size - 1) / (1 + delta) * np.eye(
                self.ensemble_size
            ) + dYb_u.T @ R_inv_sensitive @ dYb_u
            A_v = (self.ensemble_size - 1) / (1 + delta) * np.eye(
                self.ensemble_size
            ) + dYb_v.T @ R_inv_sensitive @ dYb_v
            Pa_fluc_u = LA.inv(A_u)
            Pa_fluc_v = LA.inv(A_v)
            xa_mean_u = xb_mean_u + dXb_u @ Pa_fluc_u @ dYb_u.T @ R_inv_sensitive @ (
                obs_u - H @ xb_mean_u
            )
            xa_mean_v = xb_mean_v + dXb_v @ Pa_fluc_v @ dYb_v.T @ R_inv_sensitive @ (
                obs_v - H @ xb_mean_v
            )
            values_u, vectors_u = LA.eigh(A_u)
            values_v, vectors_v = LA.eigh(A_v)
            L_u = np.diag(values_u)
            L_v = np.diag(values_v)
            dXa_u = (
                np.sqrt(self.ensemble_size - 1)
                * dXb_u
                @ vectors_u
                @ np.sqrt(LA.inv(L_u))
                @ vectors_u.T
            )
            dXa_v = (
                np.sqrt(self.ensemble_size - 1)
                * dXb_v
                @ vectors_v
                @ np.sqrt(LA.inv(L_v))
                @ vectors_v.T
            )
            dXa_u = dXa_u.T
            dXa_v = dXa_v.T
            xa_u[:, idx] = xa_mean_u[p] + dXa_u[:, p]
            xa_v[:, idx] = xa_mean_v[p] + dXa_v[:, p]

        return xa_u, xa_v

    def run(self, p_range):
        self.make_observation_matrix()
        self.make_localization_matrix(5)

        delta = 0.1
        x_u, x_v = self.parallel_analysis(p_range, delta)
        start_idx = p_range.start
        end_idx = p_range.stop - 1
        self.save_data(
            f"windvalues_analysis_{self.noise}{start_idx}_{end_idx}", {"u_x": x_u, "u_y": x_v}
        )
        self.save_data(
            f"windvalues_analysis_mean_{self.noise}{start_idx}_{end_idx}",
            {"u_x_mean": np.mean(x_u, axis=0), "u_y_mean": np.mean(x_v, axis=0)},
        )

    def plt_imshow(self, data):
        plt.imshow(data, origin="lower")
        plt.savefig(f"{self.data.PATH}/results/img/tmp.png")

    def save_data(self, filename, data):
        if isinstance(data, dict):
            # 辞書型の場合、np.savezを使用
            np.savez(f"{self.data.PATH}/results/analysis/{filename}.npz", **data)
        else:
            # NumPy型の場合、np.saveを使用
            np.save(f"{self.data.PATH}/results/analysis/{filename}.npy", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LETKF analysis.")
    parser.add_argument("p_range_start", type=int, help="Start index of p_range")
    parser.add_argument("p_range_end", type=int, help="End index of p_range")
    args = parser.parse_args()
    p_range = range(args.p_range_start, args.p_range_end + 1)
    letkf = LETKF()
    letkf.run(p_range)

