import numpy as np
import matplotlib.pyplot as plt

def show_prior(number_of_prior):
    wind_values = np.load(f'/data10/kinuki/da_rankine/data/windvalues_prior{number_of_prior}.npy')
    u_x = wind_values[0, 0]
    u_y = wind_values[0, 1]
    u = wind_values[0, 2]
    
    # 91x91の格子点を作成(for 等高線を描く)
    xx,yy = np.meshgrid(np.linspace(0, 90, 91), np.linspace(0, 90, 91))

    # 矢印の大きさをvaluesの値に比例させる
    arrow_scale = u[::5, ::5] / np.max(u)

    # プロット
    fig, ax = plt.subplots()
    im = ax.imshow(u, extent=[-1, 91, -1, 91], origin='lower', cmap='jet')
    cbar = fig.colorbar(im, ax=ax, label='Value')
    ax.set_title('Grid Point Values')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 矢印を表示する格子点の座標を取得
    arrow_x = xx[::5, ::5]
    arrow_y = yy[::5, ::5]

    # 矢印
    ax.quiver(arrow_x, arrow_y, u_x[::5, ::5], u_y[::5, ::5],scale = 5.5,angles='xy',scale_units='xy',pivot='mid')
    cont = plt.contour(xx, yy, u, levels=[14], colors='black')
    plt.savefig(f'data10/kinuki/da_rankine/results/img/prior{number_of_prior}.png')

if __name__ == '__main__':
    show_prior(number_of_prior = 1)