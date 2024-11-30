def rungeKutta(y0:float, tspan:tuple, delta_t:float, ode:callable):

    t0, tend = tspan
    n = int((tend - t0)/delta_t) 

    y = [y0]
    t = [t0]

    for i in range(1, n + 1):

        k1 = delta_t * ode(t0, y[i-1])
        k2 = delta_t * ode(t0 + 0.5 * delta_t, y[i-1] + 0.5 * k1)
        k3 = delta_t * ode(t0 + 0.5 * delta_t, y[i-1] + 0.5 * k2)
        k4 = delta_t * ode(t0 + delta_t, y[i-1] + k3)
 
        # Update next value of y
        y.append(y[i-1] + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4))
 
        # Update next value of x
        t0 = t0 + delta_t
        t.append(t0)

    return t, y