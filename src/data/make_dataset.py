import numpy as np

def wind_speed(xx, yy, x0=45, y0=45, U=30, R=12):
    dx = xx - x0
    dy = yy - y0
    r = np.sqrt(dx**2 + dy**2)
    u = np.where(r <= R, U * r / R, U * R / r)
    u_dx = np.where(r != 0, -u * dy / r, 0)
    u_dy = np.where(r != 0, u * dx / r, 0)
    return u_dx, u_dy, u

def make_Rankine_prior(xx, yy, number_of_prior, params_true, emsemble_size=60):
    if number_of_prior == 1:
        ratio = 0.25
    elif number_of_prior == 2:
        ratio = 0.5
    elif number_of_prior == 3:
        ratio = 1.0

    std_dev = ratio * params_true['R']

    params_prior = np.zeros((emsemble_size, 4))
    params_prior[:, 0] = np.random.normal(params_true['x0'], std_dev, emsemble_size)
    params_prior[:, 1] = np.random.normal(params_true['y0'], std_dev, emsemble_size)
    params_prior[:, 2] = np.repeat(params_true['U'], emsemble_size)
    params_prior[:, 3] = np.random.normal(params_true['R'], std_dev, emsemble_size)

    wind_values = np.zeros((emsemble_size, 3, xx.shape[0], xx.shape[1]))
    for i in range(emsemble_size):
        u_dx, u_dy, u = wind_speed(xx, yy, *params_prior[i])
        wind_values[i, 0] = u_dx
        wind_values[i, 1] = u_dy
        wind_values[i, 2] = u
    np.save(f'/data10/kinuki/da_rankine/data/windvalues_prior{number_of_prior}.npy', wind_values)




def main(number_of_prior):
    xx, yy = np.meshgrid(np.linspace(0, 90, 91), np.linspace(0, 90, 91))
    params_true = {'x0': 45, 'y0': 45, 'U': 30, 'R': 12}
    emsemble_size = 60
    make_Rankine_prior(xx, yy, number_of_prior, params_true, emsemble_size)

if __name__ == '__main__':
    main(number_of_prior = 1)