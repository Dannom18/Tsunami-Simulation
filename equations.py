import numpy as np

def ode(a, b, delta_x):

    da_dt = np.zeros(len(a))
    db_dt = np.zeros(len(b))

    for i in range(1, len(a) - 1):

        da_dt[i] = (-1 / (2 * delta_x)) * (a[i] * (b[i + 1] - b[i - 1]) + b[i] * (a[i + 1] - a[i - 1]))
        db_dt[i] = (-1 / (2 * delta_x)) * (9.81 * (a[i + 1] - a[i - 1]) + b[i] * (b[i + 1] - b[i - 1]))

    return da_dt, db_dt
