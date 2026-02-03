import sympy as sym
import numpy as np
import yaml


def rho(x, y, theta):
    return sym.Matrix([
        [sym.cos(theta), -sym.sin(theta), x],
        [sym.sin(theta), sym.cos(theta), y],
        [0, 0, 1]
    ])


def Ad(M):
    return sym.Matrix([
        [M[0, 0], M[0, 1], M[1, 2]],
        [M[1, 0], M[1, 1], -M[0, 2]],
        [0, 0, 1]
    ])


def metric2algebra(gamma, g):
    """
    Docstring for metric2algebra

    :param gamma: function (path) t -> SE(2)
    :param g: 3x3 matrix representing the metric on se(2)
    """
    t, a = sym.symbols('t a')
    expr = Ad(gamma(t)).inv().T @ g @ Ad(gamma(t)).inv()
    mm = sym.integrate(expr, (t, a, 1))
    return sym.Lambda(a, mm)


def inner_product(gamma, gab, xi1, xi2):
    t, s = sym.symbols('t s')

    gd1 = Ad(gamma(s)).inv() @ sym.integrate(Ad(gamma(t)) @ xi1, (t, 0, s))
    gd2 = Ad(gamma(s)).inv() @ sym.integrate(Ad(gamma(t)) @ xi2, (t, 0, s))

    return float(sym.integrate((gd1.T @ gab @ gd2)[0, 0], (s, 0, 1)))


def ortho_bases(gamma, gab, n):
    t = sym.symbols('t')
    bases = []
    for i in range(n):
        pi = t ** i
        # Gram-Schmidt
        for pj in bases:
            xj = sym.Matrix([0, 0, pj])
            proj = inner_product(gamma, gab, sym.Matrix([0, 0, pi]), xj) * pj
            pi = pi - proj

        # normalize
        xi = sym.Matrix([0, 0, pi])
        norm = sym.sqrt(inner_product(gamma, gab, xi, xi))
        bases.append(pi / norm)
    return bases


def delta_coeffs(gamma, gab, basis):
    t, s, x = sym.symbols('t s x')

    coeffs = []
    for i, bi in enumerate(basis):
        xi = sym.Matrix([0, 0, bi])
        gd1 = Ad(gamma(s)).inv() @ sym.integrate(Ad(gamma(t)) @ xi, (t, 0, s))
        gd2_a = sym.Matrix([0, 0, 0])     # Valid [0, x]
        gd2_b = sym.Matrix([0, s - x, 1]) # Valid [x, 1]

        # Integral_0^a [gd1 . gd2] = Int_x^1 [gd1 . gd2_b]
        coef = sym.integrate((gd1.T @ gab @ gd2_b)[0, 0], (s, x, 1))
        coeffs.append(sym.simplify(coef))
    return coeffs


# def to_shape(poly, tt):
#     s, t = sym.symbols('s t')
#     theta = sym.integrate(poly, (t, 0, s))
#     x = sym.Integral(sym.cos(theta), (s, 0, t))
#     y = sym.Integral(sym.sin(theta), (s, 0, t))
#     th_func = sym.lambdify(s, theta, 'numpy')

#     traj = []
#     for ti in tt:
#         xi = x.subs({t: ti})
#         yi = y.subs({t: ti})
#         traj.append((xi.evalf(), yi.evalf(), th_func(ti)))

#     traj = np.array(traj)
#     return traj


if __name__ == '__main__':
    recalc_polys = True

    t = sym.symbols('t')
    def gamma(t): return rho(t, 0, 0)
    gab = np.diag([1, 1, 0])

    sfx = '0' # [1,1,0] -> '', [1,1,1] -> '1'

    if recalc_polys:
        pp = ortho_bases(gamma, gab, 10)
        print('Calculated bases')
        polys_string = [sym.srepr(pi) for pi in pp]
        with open(f'_orthopoly{sfx}.yaml', 'w') as f:
            yaml.dump(polys_string, f)

        coeffs = delta_coeffs(gamma, gab, pp)
        coeffs_string = [sym.srepr(ci) for ci in coeffs]
        with open(f'_polycoeffs{sfx}.yaml', 'w') as f:
            yaml.dump(coeffs_string, f)
        print('Calculated coeffs')


    with open('_orthopoly.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp = [sym.sympify(ps) for ps in polys_string]
    print('loaded bases')

    with open('_polycoeffs.yaml', 'r') as f:
        coeffs_string = yaml.safe_load(f)
    coeffs = [sym.sympify(cs) for cs in coeffs_string]
    print('loaded coeffs')


    import matplotlib.pyplot as plt
    tt = np.linspace(0, 1, 100)
    ppt = np.array([[float(pi.subs({t: ti})) for ti in tt] for pi in pp])
    for i in range(5):
        plt.plot(tt, ppt[i])
    plt.legend([f'poly {i}' for i in range(5)])
    plt.show()

    exit(0)

    # print(coeffs)
    # print(np.sum(coeffs ** 2))

    def actual_traj(t):
        if t < .3:
            return [t, 0, 0]
        else:
            s = t - .3
            return np.array([.3, 0, 1]) + s * np.array([np.cos(1), np.sin(1), 0])
    traj_truth = np.array([actual_traj(ti) for ti in np.linspace(0, 1, 100)])

    poly_appx = coeffs @ pp
    traj = to_shape(poly_appx, np.linspace(0, 1, 100))
    import matplotlib.pyplot as plt
    # plt.plot(traj_truth[:, 0], traj_truth[:, 1])
    # plt.plot(traj[:, 0], traj[:, 1], c='C1')
    # # plt.plot(traj2[:, 0], traj2[:, 1])
    # plt.axis('equal')
    tt = np.linspace(0, 1, 100)
    ppt = np.array([[float(pi.subs({t: ti})) for ti in tt] for pi in pp])
    for i in range(5):
        plt.plot(tt, ppt[i])
    plt.legend([f'poly {i}' for i in range(5)])
    plt.show()
    exit(0)


    tt = np.linspace(0, 1, 100)
    poly_vals = sym.lambdify(t, poly_appx, 'numpy')(tt)

    # import matplotlib.pyplot as plt
    # # plt.plot(tt, poly_vals)
    # # plt.show()

    # s = sym.symbols('s')
    # xi = sym.Matrix([0, 0, poly_appx])
    # gdot = Ad(gamma(s)).inv() @ sym.integrate(Ad(gamma(t)) @ xi, (t, 0, s))
    # gdot_func = sym.lambdify(s, gdot, 'numpy')
    # gd_vals = np.array([gdot_func(ti) for ti in tt])[:, :, 0]

    # print(gd_vals.shape)

    # plt.plot(tt, gd_vals[:, 1])
    # plt.plot(tt, gd_vals[:, 2])
    # plt.show()
