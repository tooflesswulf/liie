import diffgeo_nd as dn
import npoly2d as n2

import sympy as sym
import numpy as np
import yaml

def characteristic(ll, metric, N=100):
    mesh_points = np.array([
        [-ll, -ll],
        [ ll, -ll],
        [-ll,  ll],
        [ ll,  ll]
    ])
    mesh_simplices = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])
    eps = 1e-5

    mesh_domain = {'type': 'mesh', 'points': mesh_points, 'simplices': mesh_simplices}
    chi, err = dn.euler_characteristic(metric, mesh_domain, num_samples=500)
    print(f'chi: {chi}, error: {err}')



if __name__ == '__main__':
    N = 100

    # Arm configuration
    _gamma = np.zeros((N, 3))
    _gamma[:, 0] = np.linspace(0, 1, N)  # Straight arm along x-axis
    gamma = np.array([n2.rho(gi) for gi in _gamma])

    sx = sym.symbols('t')
    sfx = ''  # [1,1,0] -> '', [1,1,1] -> '1'
    if sfx == '':
        gab = np.diag([1, 1, 0])
    elif sfx == '1':
        gab = np.diag([1, 1, 1])
    
    # Load shape basis functions
    with open(f'_orthopoly{sfx}.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp = [sym.sympify(ps) for ps in polys_string]
    print('loaded bases')

    def eval_poly(x):
        return np.array([pi.subs({sx: x}) for pi in pp]).astype(float)
    tt = np.linspace(0, 1, 100)
    npoly = np.array([eval_poly(ti) for ti in tt])

    bases = np.zeros((npoly.shape[1], N, 3))
    for i in range(npoly.shape[1]):
        bases[i, :, 2] = npoly[:, i]

    def metric(xx):
        x, y = xx
        gamma = n2.shape_exp(x * bases[0] + y * bases[1])
        g00 = n2.inner_product(gamma, gab, bases[0], bases[0])
        g01 = n2.inner_product(gamma, gab, bases[0], bases[1])
        g11 = n2.inner_product(gamma, gab, bases[1], bases[1])
        # print(g01, n2.inner_product(gamma, gab, bases[1], bases[0]))
        return np.array([[g00, g01], [g01, g11]])

    print('computing characteristic')
    characteristic(.6, metric)

