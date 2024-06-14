import matplotlib.pyplot as plt
import numpy as np

from src.config import config
from src.data.dataset import RankineData


def show_prior(number_of_prior):
    point = 0
    data = RankineData(number_of_prior)
    u_x, u_y, u = (
        data.windvalues_prior["u_x"],
        data.windvalues_prior["u_y"],
        data.windvalues_prior["u"],
    )

    # 91x91の格子点を作成(for 等高線を描く)
    xx, yy = data.xx, data.yy

    # プロット
    fig, ax = plt.subplots()
    im = ax.imshow(u[point], extent=[-1, 91, -1, 91], origin="lower", cmap="jet")
    cbar = fig.colorbar(im, ax=ax, label="Value")
    ax.set_title("Grid Point Values")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # 矢印を表示する格子点の座標を取得
    arrow_x = xx[::5, ::5]
    arrow_y = yy[::5, ::5]

    # 矢印
    ax.quiver(
        arrow_x,
        arrow_y,
        u_x[point][::5, ::5],
        u_y[point][::5, ::5],
        scale=5.5,
        angles="xy",
        scale_units="xy",
        pivot="mid",
    )
    cont = plt.contour(xx, yy, u, levels=[14], colors="black")
    plt.savefig(f"data10/kinuki/da_rankine/results/img/prior{number_of_prior}.png")


def show_analysis():
    data = np.load(
        "/data10/kinuki/da_rankine/results/analysis/combined_windvalues_analysis.npz",
        allow_pickle=True,
    )
    xx, yy = np.meshgrid(np.linspace(0, 90, 91), np.linspace(0, 90, 91))
    u_x = data["u_x"]
    u_y = data["u_y"]
    u = np.sqrt(u_x**2 + u_y**2)
    u = u.reshape(config.ensemble_size, 91, 91)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)

    cmaps = ["Red", "Blue", "Green", "Orange", "Purple"]
    for i in range(config.ensemble_size):
        ax.contour(xx, yy, u[i], levels=[14], colors=cmaps[i % 5])
    plt.savefig(f"/data10/kinuki/da_rankine/results/img/analysis.png")


if __name__ == "__main__":
    # show_prior(number_of_prior = 1)
    show_analysis()
