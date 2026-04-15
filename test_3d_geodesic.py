import numpy as np
import sympy as sym
import yaml


def load_coeffs(file):
    with open(file, 'r') as f:
        coeffs_string = yaml.safe_load(f)
    coeffs = [sym.sympify(cs) for cs in coeffs_string]

    def eval_coeffs(x):
        return np.array([ci.subs({sym.symbols('x'): x}) for ci in coeffs]).astype(float)
    return eval_coeffs


if __name__ == '__main__':
    sx = sym.symbols('x')

    # [1,1,0] -> '', [1,1,1] -> '1', [0,0,1] -> '2'
    with open(f'_orthopoly.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp0 = [sym.sympify(ps) for ps in polys_string]
    coeff0 = load_coeffs('_polycoeffs.yaml')

    with open(f'_orthopoly1.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp1 = [sym.sympify(ps) for ps in polys_string]
    coeff1 = load_coeffs('_polycoeffs1.yaml')

    with open(f'_orthopoly2.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp2 = [sym.sympify(ps) for ps in polys_string]
    coeff2 = load_coeffs('_polycoeffs2.yaml')

    gab = np.diag([1, 1, 1, 0, 0, 0])

    def axes2coeffs(ll, axes):
        lens = np.cumsum(ll)
        for li, axi in zip(lens, axes):
            if axi == (1, 0, 0):
                yield np.r_[coeff0(li), z, z]
            elif axi == (0, 1, 0):
                yield np.r_[z, coeff0(li), z]
            elif axi == (0, 0, 1):
                yield np.r_[z, z, coeff2(li)]
            else:
                raise ValueError(f"Unsupported axis {axi}")


    ll1 = [0, .33, .33, .33]
    axes1 = [
        (0, 0, 1),  # Joint 1: Revolute around Z-axis
        (0, 1, 0),  # Joint 2: Revolute around Y-axis
        (1, 0, 0),  # Joint 3: Revolute around X-axis
        (0, 0, 1),  # Joint 4: Revolute around Z-axis
    ]

    ll2 = [0, .5, .5]
    axes2 = [
        (0, 0, 1),
        (1, 0, 0),
        (0, 0, 1),
    ]
