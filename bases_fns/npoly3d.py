import numpy as np
import pinocchio as pin
from scipy import integrate

import matplotlib.pyplot as plt


def gammadot(gamma, xi):
    # gammadot(s) = Ad(gamma(s)).inv() @ int_0^s Ad(gamma(t)) @ xi dt
    N = gamma.shape[0]

    Ad_gamma_t = np.array([pin.SE3(gamma[ti]).action for ti in range(N)])
    Ad_gammainv_t = np.array([pin.SE3(gamma[ti]).inverse().action for ti in range(N)])
    integrand = Ad_gamma_t @ xi[:,:,None]

    cum_integral = integrate.cumulative_trapezoid(integrand, dx=1 / (N - 1), axis=0, initial=0)
    return (Ad_gammainv_t @ cum_integral)[:, :, 0]


def inner_product(gamma, gab, xi1, xi2):
    N = gamma.shape[0]
    gdot1 = gammadot(gamma, xi1)
    gdot2 = gammadot(gamma, xi2)

    # Compute the inner product as an integral over s
    integrand = np.einsum('si,ij,sj->s', gdot1, gab, gdot2)
    return np.trapezoid(integrand, dx=1 / (N - 1))


def metric(gamma, gab, xi_bases):
    gdot_bases = [gammadot(gamma, xi) for xi in xi_bases]



def shape_exp(xi):
    # Given twist trajectory xi(s), integrate to get SE(3) trajectory gamma(s)
    N = xi.shape[0]
    xi = xi + [0, 0, 1, 0, 0, 0]  # Add identity component for integration
    gi = np.eye(4)
    gamma = [gi]
    for i in range(1, N):
        accu = pin.exp(xi[i - 1] * (1 / (N - 1)))
        gi = gi @ accu
        gamma.append(gi)
    return np.array(gamma)


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
