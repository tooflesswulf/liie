import sympy as sym
import numpy as np
import yaml

import matplotlib.pyplot as plt


def arm_pts(qs, jnt_locs):
    l0 = jnt_locs[0]
    ll = np.diff(np.hstack((jnt_locs, [1])))

    pts = [(0, 0), (l0, 0)]
    for i in range(len(ll)):
        x, y = pts[-1]
        angle = np.sum(qs[:i + 1])
        x += ll[i] * np.cos(angle)
        y += ll[i] * np.sin(angle)
        pts.append((x, y))

    pts = np.array(pts)
    return pts

if __name__ == '__main__':
    sx = sym.symbols('x')
    sfx = '' # [1,1,0] -> '', [1,1,1] -> '1'

    with open(f'_orthopoly{sfx}.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp = [sym.sympify(ps) for ps in polys_string]
    print('loaded bases')

    with open(f'_polycoeffs{sfx}.yaml', 'r') as f:
        coeffs_string = yaml.safe_load(f)
    coeffs = [sym.sympify(cs) for cs in coeffs_string]
    print('loaded coeffs')

    def eval_coeffs(x):
        return np.array([ci.subs({sx: x}) for ci in coeffs]).astype(float)

    # Robot1: 2-link robot w/ joints at [0, .5]
    m1 = np.array([eval_coeffs(0), eval_coeffs(.5)]).T

    # Robot2: 3-link robot w/ joints at [0, .33, .66]
    m2 = np.array([eval_coeffs(0), eval_coeffs(.33), eval_coeffs(.66)]).T

    # Make interactive sliders for joint angles
    plt.ion()
    # j1 = [.5, -.5]
    j1 = [1, -2]
    xy = [0.938791, 0.239713]
    j2 = np.linalg.pinv(m2) @ m1 @ j1
    # j2 = [23/20, -23/20, -23/20]

    a1 = arm_pts(j1, [0, .5])
    a2 = arm_pts(j2, [0, .33, .66])
    print('j1:', j1)
    print('j2:', j2)

    plt.figure(1)
    tt = np.linspace(0, 1, 100)
    rob1 = np.c_[tt, np.maximum(0, tt - .5)]
    rob2 = np.c_[tt, np.maximum(0, tt - .33), np.maximum(0, tt - .66)]
    gd1 = rob1 @ j1
    gd2 = rob2 @ j2
    pgd1, = plt.plot(tt, gd1)
    pgd2, = plt.plot(tt, gd2)
    plt.ylim(-1, 1)

    fig, ax = plt.subplots()
    r1, = plt.plot(*a1.T, '.-')
    r2, = plt.plot(*a2.T, '.-')
    pt = plt.scatter(*xy, marker='x', c='r')
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

    while True:
    #     ax.clear()
        qs = plt.ginput(n=1, timeout=0)
        if len(qs) < 1:
            break
        xy = np.array(qs[0])
        pt.set_offsets(xy)

        dd = np.clip(np.linalg.norm(xy), 0, 1)
        th = np.arctan2(xy[1], xy[0])
        phi = np.arccos(dd)

        j1 = [th + phi, -2*phi]
        j2 = np.linalg.pinv(m2) @ m1 @ j1
        print('j1:', j1)
        print('j2:', j2)

        a1 = arm_pts(j1, [0, .5])
        a2 = arm_pts(j2, [0, .33, .66])
        r1.set_data(a1.T)
        r2.set_data(a2.T)

        gd1 = rob1 @ j1
        gd2 = rob2 @ j2
        pgd1.set_data(np.c_[tt, gd1].T)
        pgd2.set_data(np.c_[tt, gd2].T)

        plt.pause(1 / 20)
