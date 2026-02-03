import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def hat(xi):
    # Return matrix representation of se(2) element
    return np.array([
        [0, -xi[2], xi[0]],
        [xi[2], 0, xi[1]],
        [0, 0, 0]
    ])

def integrate_discrete(curve, dt):
    M = np.eye(3)
    trajectory = []

    for xihat in curve:
        M = M @ expm(hat(xihat * dt))
        trajectory.append(M)
    return np.array(trajectory)


def Ad(M):
    return np.array([
        [M[0, 0], M[0, 1], M[1, 2]],
        [M[1, 0], M[1, 1], -M[0, 2]],
        [0, 0, 1]
    ])

if __name__ == "__main__":
    t = np.linspace(0, 1, 1000)
    dt = t[1] - t[0]
    z = np.zeros_like(t)
    e = np.c_[z+1, z, z]

    # print(e)

    n = 2
    c1 = e + np.c_[z, z, z + 1 / n]

    id = integrate_discrete(e, dt)
    g1 = integrate_discrete(c1, dt)
    # g1 = integrate_discrete(e, dt)

    th = np.sin(t / 4 / np.pi)
    ideal = np.array([
        [np.cos(t/n), -np.sin(t/n), n*np.sin(t/n)],
        [np.sin(t/n), np.cos(t/n), n-n*np.cos(t/n)],
        [z, z, z+1]
    ]).transpose((2, 0, 1))
    print('dist 2 ideal', np.linalg.norm(ideal - g1))

    # Test perturbation
    eps = .0001
    xi = np.c_[z, z, np.cos(3 * np.pi * t)]
    g2 = integrate_discrete(c1 + eps * xi, dt)
    # g2 = integrate_discrete(e + eps * xi, dt)
    xyq1 = np.c_[g1[:, 0, 2], g1[:, 1, 2], np.arctan2(g1[:, 1, 0], g1[:, 0, 0])]
    xyq2 = np.c_[g2[:, 0, 2], g2[:, 1, 2], np.arctan2(g2[:, 1, 0], g2[:, 0, 0])]
    dist2 = np.sqrt(np.sum((xyq2 - xyq1) ** 2) / len(t))
    print('dist 2 perturbed', dist2 / eps)
    # print('dist 2 perturbed', np.linalg.norm(g2 - g1) / eps)

    # Pertubation v2 (Adjoints)
    vels = np.array([Ad(M) @ xx for M, xx in zip(g1, xi)])
    vels_accu = np.cumsum(vels * dt, axis=0)
    g1_dot = np.array([np.linalg.inv(Ad(gi)) @ v for gi, v in zip(g1, vels_accu)])
    # print((xyq2 - xyq1) / eps)
    # print(g1_dot)
    exit(0)

    # nn = 20
    # plt.plot(t, z)
    # plt.quiver(t[::nn], z[::nn], z[::nn], xi[::nn, 2])
    # plt.show()
    # exit(0)

    # plt.plot(id[:, 0, 2], id[:, 1, 2])
    plt.plot(g1[:, 0, 2], g1[:, 1, 2])
    # plt.plot(ideal[:, 0, 2], ideal[:, 1, 2])
    plt.plot(g2[:, 0, 2], g2[:, 1, 2])
    nn = 20
    plt.quiver(g1[::nn, 0, 2], g1[::nn, 1, 2], g1_dot[::nn, 0], g1_dot[::nn, 1], color='red', scale=.5)
    plt.axis('equal')
    plt.show()
