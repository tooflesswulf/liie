import numpy as np
import pinocchio as pin

import matplotlib.pyplot as plt

def gdot(gamma, delta, axis):
    N = gamma.shape[0]
    ix = delta * (N - 1)

    # Interpolate gamma at the point corresponding to delta
    # For simplicity, we can use linear interpolation here
    i0 = int(np.floor(ix))
    i1 = min(i0 + 1, N - 1)
    alpha = ix - i0
    gamma_delta = (1 - alpha) * gamma[i0] + alpha * gamma[i1]

    # Adjoint transform of the interpolated gamma on the axis
    int_adg = pin.exp(gamma_delta).action @ axis
    gdots = []
    for i, gi in enumerate(gamma):
        if i < ix:
            gdots.append(np.zeros(6))
            continue
        gdots.append(pin.exp(gi).inverse().action @ int_adg)
    gdots = np.array(gdots)
    return gdots


def inner_product(gamma, gab, delta1, delta2):
    gdot1 = gdot(gamma, delta1, [0, 0, 0, 1, 0, 0])
    # N = gamma.shape[0]
    # for i in range(N):
    #     t = i / (N - 1)
    #     Ad_gamma = pin.exp(gamma[i]).action
    #     break

    print(gdot1)
    exit(0)
    # for i in range(len())
    # gd1 = Ad(gamma(s)).inv() @ sym.integrate(Ad(gamma(t)) @ delta1, (t, 0, s))
    pass


if __name__ == '__main__':
    N = 100

    # Arm configuration
    gamma = np.zeros((N, 6))
    gamma[:, 2] = np.linspace(0, 1, N)  # Straight arm along z-axis

    print(gamma[1,2] - gamma[0,2], 1/(N-1))

    gab = np.diag([1, 1, 1, 0, 0, 0])  # Inner product matrix

    inner_product(gamma, gab, .2, 0)
    exit(0)

    # M = pin.exp(gamma)
    # print(M)
    # exit(0)

    ax = plt.gcf().add_subplot(projection='3d')
    ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2])
    ax.set_title("Initial Arm Configuration")
    plt.show()
