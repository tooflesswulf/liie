import sympy as sym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def diag_integrals(gamma, met):
    # Integrals for diagonal sections. Need to be split in to simplify max(x, y).
    i, n = sym.symbols('i n')
    x, y = sym.symbols('x y')

    expr = Ad(gamma(x)).T @ met(x) @ Ad(gamma(y))

    # (i+1, i+1) contribution
    sym11 = sym.integrate(expr * (n * x - i) * (n * y - i), (y, i / n, x), (x, i / n, (i + 1) / n))
    sym11 = sym11 + sym11.T

    sym00 = sym.integrate(expr * (i + 1 - n * x) * (i + 1 - n * y), (y, i / n, x), (x, i / n, (i + 1) / n))
    sym00 = sym00 + sym00.T

    # Should be Ad.T @ met(y) @ Ad for 2nd integral, but I'm lazy and swap x,y.
    sym01 = sym.integrate(expr * (i + 1 - n * x) * (n * y - i), (y, i / n, x), (x, i / n, (i + 1) / n)) + \
        sym.integrate(expr * (n * x - i) * (i + 1 - n * y), (y, i / n, x), (x, i / n, (i + 1) / n)).T
    return sym00, sym01, sym11


def off_integrals(gamma, met):
    # Integrals for off-diagonal sections, assuming i > j.
    i, j, n = sym.symbols('i j n')
    x, y = sym.symbols('x y')

    expr = Ad(gamma(x)).T @ met(x) @ Ad(gamma(y))

    asym00 = sym.integrate(expr * (i + 1 - n * x) * (j + 1 - n * y), (y, j / n, (j + 1) / n), (x, i / n, (i + 1) / n))
    asym01 = sym.integrate(expr * (i + 1 - n * x) * (n * y - j), (y, j / n, (j + 1) / n), (x, i / n, (i + 1) / n))
    asym10 = sym.integrate(expr * (n * x - i) * (j + 1 - n * y), (y, j / n, (j + 1) / n), (x, i / n, (i + 1) / n))
    asym11 = sym.integrate(expr * (n * x - i) * (n * y - j), (y, j / n, (j + 1) / n), (x, i / n, (i + 1) / n))
    return asym00, asym01, asym10, asym11


def area_elements(m, gamma, met):
    fm = float(m)
    i, j, n = sym.symbols('i j n')

    d00, d01, d11 = diag_integrals(gamma, met)
    o00, o01, o10, o11 = off_integrals(gamma, met)

    diags = {
        'pp': sym.lambdify(i, d00.subs({n: fm}), 'numpy'),
        'pn': sym.lambdify(i, d01.subs({n: fm}), 'numpy'),
        'np': sym.lambdify(i, d01.subs({n: fm}).T, 'numpy'),
        'nn': sym.lambdify(i, d11.subs({n: fm}), 'numpy'),
    }
    offdiags = {
        'pp': sym.lambdify((i, j), o00.subs({n: fm}), 'numpy'),
        'pn': sym.lambdify((i, j), o01.subs({n: fm}), 'numpy'),
        'np': sym.lambdify((i, j), o10.subs({n: fm}), 'numpy'),
        'nn': sym.lambdify((i, j), o11.subs({n: fm}), 'numpy'),
    }

    def area_index(i, j, ix):
        if i == j:
            return diags[ix](i)
        elif i > j:
            return offdiags[ix](i, j)
        else:
            return offdiags[ix[::-1]](j, i).T

    result = []
    for i in tqdm(range(m + 1)):
        row = []
        for j in range(m + 1):
            o = []
            if i < m and j < m:
                o.append(area_index(i, j, 'pp'))
            if i < m and j > 0:
                o.append(area_index(i, j - 1, 'pn'))
            if i > 0 and j < m:
                o.append(area_index(i - 1, j, 'np'))
            if i > 0 and j > 0:
                o.append(area_index(i - 1, j - 1, 'nn'))
            out = sum(o, start=sym.zeros(3, 3))
            row.append(np.array(out).astype(np.float64))
        result.append(row)

    area_mat = np.block(result)
    return area_mat


def ad_gamma(m, gamma):
    i, n = sym.symbols('i n')
    x = sym.symbols('x')

    expr1 = sym.integrate(Ad(gamma(x)) * (i + 1 - n * x), (x, i / n, (i + 1) / n))
    expr2 = sym.integrate(Ad(gamma(x)) * (n * x - i), (x, i / n, (i + 1) / n))
    # f1 = sym.Lambda(i, expr1.subs({n: float(m)}))
    f1 = sym.lambdify(i, expr1.subs({n: float(m)}), 'numpy')
    # f2 = sym.Lambda(i, expr2.subs({n: float(m)}))
    f2 = sym.lambdify(i, expr2.subs({n: float(m)}), 'numpy')

    result = []
    for i in tqdm(range(m + 1)):
        row = []
        for j in range(m + 1):
            item = sym.zeros(3, 3)
            if i > j:
                item += f1(j)
            if j > 0 and i > j - 1:
                item += f2(j - 1)
            row.append(np.array(Ad(gamma(i / float(m))).inv() @ item).astype(np.float64))
        result.append(row)

    ad_mat = np.block(result)
    return ad_mat


def laplacian(m):
    L = -2 * np.eye(3 * (m + 1)) + np.eye(3 * (m + 1), k=3) + np.eye(3 * (m + 1), k=-3)
    L[:3, :3] = -np.eye(3)
    L[-3:, -3:] = -np.eye(3)
    return L * m**2


if __name__ == '__main__':
    t = sym.symbols('t')
    def gamma(t): return rho(t, 0, 0)
    met = metric2algebra(gamma, sym.eye(3))

    nn = 100

    AA = ad_gamma(nn, gamma)
    print('Computed AA')
    # np.save('A50.npy', AA)
    # AA = np.load('A50.npy')

    M = area_elements(nn, gamma, met)
    print('Computed M')
    # np.save('M50.npy', M)
    # M = np.load('M50.npy')

    brr = np.ones(len(M))
    brr[0] = brr[-1] = 1/2
    t_int = np.diag(brr) / nn

    M2 = AA.T @ t_int @ AA


    t = np.linspace(0, 1, nn + 1)
    xi = np.c_[0 * t, 0 * t, np.sin(np.pi * t)].flatten()
    xi2 = np.c_[0 * t, 0 * t, 1 - (1-2*t)**2].flatten()

    Lxi = np.c_[0*t, np.sin(np.pi * t), np.pi * np.cos(np.pi * t)].flatten()
    Lxi2 = np.c_[0*t, 1 - (1-2*t)**2, 0*t - 8].flatten()

    L = laplacian(nn)

    print(xi @ M @ xi2)
    print(xi2 @ M @ xi)

    # print(xi @ M @ Lxi2)
    # print(xi2 @ M @ Lxi)
    # print(xi @ AA.T @ t_int @ L @ AA @ xi2)
    # print(xi2 @ AA.T @ t_int @ L @ AA @ xi)
    exit(0)

    # print(xi @ M2 @ xi)

    # print(xi @ M2 @ xi2)

    # Lop = np.linalg.pinv(AA) @ L @ AA

    # print(xi @ M @ Lop @ xi2)
    # print(xi @ M2 @ Lop @ xi2)

    # print(xi2 @ M @ Lop @ xi)
    # print(xi2 @ M2 @ Lop @ xi)

    exit(0)

    t = np.linspace(0, 1, 51)

    xi = np.c_[0 * t, 0 * t, np.sin(2 * np.pi * t)].flatten()
    xi2 = np.c_[0 * t, 0 * t, np.sin(t)].flatten()

    # "Hermitian"-ify L
    print(xi @ M @ xi2)
    print(xi2 @ M @ xi)

    # Lh = (M @ L + L.T @ M) / 2
    Minv = np.linalg.inv(M)
    Lh = (Minv @ L @ M + L) / 2
    print('---')
    print(xi @ M @ Lh @ xi2)
    print(xi2 @ M @ Lh @ xi)

    print('---')
    eig2 = np.linalg.eig(Lh)
    v1 = eig2.eigenvectors[:, 0]
    v2 = eig2.eigenvectors[:, 1]
    print(v1 @ M @ v2)

    brr = v1.reshape(-1, 3)
    plt.plot(t, brr[:, 0], '.-')
    plt.plot(t, brr[:, 1], '.-')
    plt.plot(t, brr[:, 2], '.-')
    plt.show()
    exit(0)

    # a = xi.reshape(-1, 3)
    # b = (Lh @ xi).reshape(-1, 3)

    # import matplotlib.pyplot as plt
    # plt.subplot(221)
    # plt.plot(t, a[:, 0], '.-')
    # plt.plot(t, b[:, 0], '.-')
    # plt.ylim([-1, 1])
    # plt.subplot(222)
    # plt.plot(t, a[:, 1], '.-')
    # plt.plot(t, b[:, 1], '.-')
    # plt.ylim([-1, 1])
    # plt.subplot(223)
    # plt.plot(t, a[:, 2], '.-')
    # plt.plot(t, b[:, 2], '.-')
    # plt.ylim([-1, 1])

    # plt.show()

    # exit(0)

    # L = (L + L.T) / 2

    # ll = L @ xi

    # # print(L[::3, ::3])

    # eig = np.linalg.eigh(-L)
    # print(eig.eigenvalues[:10])
    # # print(eig.eigenvectors[:, 3])

    # import matplotlib.pyplot as plt
    # # plt.plot(t, -(2*np.pi)**2 * xi[2::3], '.-')
    # # plt.plot(t, ll[2::3], '.-')
    # plt.plot(t, eig.eigenvectors[0::3, 6], '.-')
    # plt.plot(t, eig.eigenvectors[1::3, 6], '.-')
    # plt.plot(t, eig.eigenvectors[2::3, 6], '.-')
    # plt.show()
    # exit(0)

    # xi = np.c_[0 * t, 0 * t, np.cos(t)]
    # xi = xi.flatten()
    # xi2 = np.c_[0 * t, 0 * t, np.sin(t)]
    # xi2 = xi2.flatten()

    # print(xi @ M @ xi2)
    # print(M.shape)
