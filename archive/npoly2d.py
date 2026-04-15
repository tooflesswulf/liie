import numpy as np
from scipy.linalg import expm


def rho(se2_vec):
    x, y, th = se2_vec
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x],
                     [s, c, y],
                     [0, 0, 1]])


def hat(se2_vec):
    x, y, th = se2_vec
    return np.array([[0, -th, x],
                     [th, 0, y],
                     [0, 0, 0]])


def Ad(SE2):
    R = SE2[:2, :2]
    tr = SE2[:2, 2]
    Ad_SE2 = np.eye(3)
    Ad_SE2[:2, :2] = R
    Ad_SE2[:2, 2] = np.array([-tr[1], tr[0]])
    return Ad_SE2


def gammadot(gamma, xi):
    # gammadot(s) = Ad(gamma(s)).inv() @ int_0^s Ad(gamma(t)) @ xi dt
    N = gamma.shape[0]
    gammadots = []

    integrand = []
    for ti in range(N):
        integrand.append(Ad(gamma[ti]) @ xi[ti])

    for ti in range(N):
        integral = np.trapezoid(integrand[:ti + 1], dx=1 / (N - 1), axis=0)

        Ad_gamma_s_inv = Ad(np.linalg.inv(gamma[ti]))
        gammadots.append(Ad_gamma_s_inv @ integral)
    return np.array(gammadots)


def inner_product(gamma, gab, xi1, xi2):
    N = gamma.shape[0]
    gdot1 = gammadot(gamma, xi1)
    gdot2 = gammadot(gamma, xi2)

    # Compute the inner product as an integral over s
    integrand = np.einsum('si,ij,sj->s', gdot1, gab, gdot2)
    return np.trapezoid(integrand, dx=1 / (N - 1))


def shape_exp(xi):
    # Given twist trajectory xi(s), integrate to get SE(2) trajectory gamma(s)
    N = xi.shape[0]
    xi = xi + [1, 0, 0]
    gi = np.eye(3)
    gamma = [gi]
    for i in range(1, N):
        accu = expm(hat(xi[i - 1]) * (1 / (N - 1)))
        gi = gi @ accu
        gamma.append(gi)
    return np.array(gamma)


if __name__ == '__main__':
    N = 100
    # Arm configuration
    _gamma = np.zeros((N, 3))
    _gamma[:, 0] = np.linspace(0, 1, N)  # Straight arm along x-axis
    gamma = np.array([rho(gi) for gi in _gamma])

    xi = np.zeros((N, 3))
    xi[:, 2] = 1.0  # Twist along theta

    brr = gammadot(gamma, xi)
    # print(inner_product(gamma, np.eye(3), xi, xi))

    import yaml
    import sympy as sym
    sx = sym.symbols('t')
    sfx = ''  # [1,1,0] -> '', [1,1,1] -> '1'
    gab = np.diag([1, 1, 0])

    with open(f'bases_fns/_orthopoly{sfx}.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp = [sym.sympify(ps) for ps in polys_string]
    print('loaded bases')

    def eval_poly(x):
        return np.array([pi.subs({sx: x}) for pi in pp]).astype(float)

    tt = np.linspace(0, 1, 100)
    npoly = np.array([eval_poly(ti) for ti in tt])

    xi = np.zeros((N, 3))
    xi[:, 2] = npoly[:, 0]
    xi2 = np.zeros((N, 3))
    xi2[:, 2] = npoly[:, 1]
    print(inner_product(gamma, gab, xi, xi2))

    gamma2 = shape_exp(xi/2 + xi2/2)
    g00 = inner_product(gamma2, gab, xi, xi)
    g01 = inner_product(gamma2, gab, xi, xi2)
    g11 = inner_product(gamma2, gab, xi2, xi2)
    print(np.array([[g00, g01], [g01, g11]]))
    # print(inner_product(gamma, gab, xi, xi))
    # print(inner_product(gamma, gab, xi, xi2))
    # print(inner_product(gamma, gab, xi2, xi2))
    # print(inner_product(gamma, gab, xi2, xi))

    xx = gamma2[:, 0, 2]
    yy = gamma2[:, 1, 2]
    import matplotlib.pyplot as plt
    plt.plot(xx, yy)
    plt.axis('equal')
    plt.show()

    # print(gammadot(gamma, xi))

    # print(eval_poly(tt[0]))
    # p0 = np.array([[float(pi.subs({sx: ti})) for ti in tt] for pi in pp])
