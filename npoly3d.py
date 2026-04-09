import numpy as np
import pinocchio as pin

import matplotlib.pyplot as plt


def gammadot(gamma, xi):
    # gammadot(s) = Ad(gamma(s)).inv() @ int_0^s Ad(gamma(t)) @ xi dt
    N = gamma.shape[0]
    gammadots = []

    integrand = []
    for ti in range(N):
        Ad_gamma_t = pin.SE3(gamma[ti]).action
        integrand.append(Ad_gamma_t @ xi[ti])

    for ti in range(N):
        integral = np.trapezoid(integrand[:ti + 1], dx=1 / (N - 1), axis=0)

        Ad_gamma_s_inv = pin.SE3(gamma[ti]).inverse().action
        gammadots.append(Ad_gamma_s_inv @ integral)
    return np.array(gammadots)


def inner_product(gamma, gab, xi1, xi2):
    N = gamma.shape[0]
    gdot1 = gammadot(gamma, xi1)
    gdot2 = gammadot(gamma, xi2)

    # Compute the inner product as an integral over s
    integrand = np.einsum('si,ij,sj->s', gdot1, gab, gdot2)
    return np.trapezoid(integrand, dx=1 / (N - 1))


if __name__ == '__main__':
    N = 100

    # Arm configuration
    _gamma = np.zeros((N, 6))
    _gamma[:, 2] = np.linspace(0, 1, N)  # Straight arm along z-axis
    gamma = np.array([np.array(pin.exp(gi)) for gi in _gamma])

    xi = np.zeros((N, 6))
    xi[:, -2] = 1.0  # Twist along x-axis

    brr = gammadot(gamma, xi)
    print(brr)

    # print(gamma[1,2] - gamma[0,2], 1/(N-1))

    # gab = np.diag([1, 1, 1, 0, 0, 0])  # Inner product matrix

    # inner_product(gamma, gab, .2, 0)
    exit(0)

    # M = pin.exp(gamma)
    # print(M)
    # exit(0)

    ax = plt.gcf().add_subplot(projection='3d')
    ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2])
    ax.set_title("Initial Arm Configuration")
    plt.show()
