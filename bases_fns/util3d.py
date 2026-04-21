import sympy as sym
import numpy as np
import yaml

import bases_fns.npoly3d as npoly3d


def load_poly(file):
    with open(file, 'r') as f:
        polys_string = yaml.safe_load(f)
    polys = sym.Array([sym.sympify(ps) for ps in polys_string])

    t = sym.symbols('t')
    poly_fns = sym.lambdify(t, polys, 'numpy')
    def np_poly(t): return np.array(poly_fns(t)).astype(float)

    return np.vectorize(np_poly, signature='()->(n)')


def load_coeffs(file):
    with open(file, 'r') as f:
        coeffs_string = yaml.safe_load(f)
    coeffs = [sym.sympify(cs) for cs in coeffs_string]

    def eval_coeffs(x):
        return np.array([ci.subs({sym.symbols('x'): x}) for ci in coeffs]).astype(float)
    return eval_coeffs


class Bases3D:
    size: int = 10
    tt: np.ndarray = None
    bases: np.ndarray = None  # shape (3*size, N, 6)

    def __init__(self, gab_diag, emb_size=10):
        self.gab = np.diag(gab_diag)
        self.size = emb_size

        # [1,1,0] -> '', [1,1,1] -> '1', [0,0,1] -> '2'
        pp0 = load_poly('bases_fns/_orthopoly.yaml')
        coeff0_fn = load_coeffs('bases_fns/_polycoeffs.yaml')

        pp1 = load_poly('bases_fns/_orthopoly1.yaml')
        coeff1_fn = load_coeffs('bases_fns/_polycoeffs1.yaml')

        pp2 = load_poly('bases_fns/_orthopoly2.yaml')
        coeff2_fn = load_coeffs('bases_fns/_polycoeffs2.yaml')

        self.x_coef_fn = None
        self.y_coef_fn = None
        self.z_coef_fn = None
        rx_sig = [gab_diag[1], gab_diag[3]]
        ry_sig = [gab_diag[0], gab_diag[4]]
        rz_sig = [gab_diag[5]]

        if rx_sig == [1, 0]:
            self.x_coef_fn = lambda x: coeff0_fn(x)
            self.x_poly_fn = lambda x: pp0(x)
        elif rx_sig == [1, 1]:
            self.x_coef_fn = lambda x: coeff1_fn(x)
            self.x_poly_fn = lambda x: pp1(x)
        else:
            raise ValueError(f"Unsupported signature for x-axis: {rx_sig}")
        if ry_sig == [1, 0]:
            self.y_coef_fn = lambda x: coeff0_fn(x)
            self.y_poly_fn = lambda x: pp0(x)
        elif ry_sig == [1, 1]:
            self.y_coef_fn = lambda x: coeff1_fn(x)
            self.y_poly_fn = lambda x: pp1(x)
        else:
            raise ValueError(f"Unsupported signature for y-axis: {ry_sig}")
        if rz_sig == [1]:
            self.z_coef_fn = lambda x: coeff2_fn(x)
            self.z_poly_fn = lambda x: pp2(x)
        else:
            raise ValueError(f"Unsupported signature for z-axis: {rz_sig}")

    def set_spacing(self, tt):
        self.tt = tt
        self.bases = self._precompute_bases(tt)

    def _precompute_bases(self, tt):
        # Build (N, 6) basis vectors for each embedding coordinate.
        # emb2xi maps x/y/z coefficients into xi columns 3/4/5 (omega_x/y/z).
        size = self.size
        x_polys = self.x_poly_fn(tt)[:, :size]
        y_polys = self.y_poly_fn(tt)[:, :size]
        z_polys = self.z_poly_fn(tt)[:, :size]

        bases = []
        for k in range(size):
            b = np.zeros((len(tt), 6))
            b[:, 3] = x_polys[:, k]
            bases.append(b)
        for k in range(size):
            b = np.zeros((len(tt), 6))
            b[:, 4] = y_polys[:, k]
            bases.append(b)
        for k in range(size):
            b = np.zeros((len(tt), 6))
            b[:, 5] = z_polys[:, k]
            bases.append(b)
        return np.array(bases)

    def embed(self, lengths, axes):
        # Given arm segment lengths and axes, compute the embedding coefficients by projecting onto the bases.
        # The embedding for the robot arm with joint angles q_i is sum_i q_i * coeffs_i
        embeds = []
        lengths_sum = np.cumsum(lengths)
        for li, axi in zip(lengths_sum, axes):
            x_emb = self.x_coef_fn(li)[:self.size]
            y_emb = self.y_coef_fn(li)[:self.size]
            z_emb = self.z_coef_fn(li)[:self.size]
            emb = np.r_[axi[0] * x_emb, axi[1] * y_emb, axi[2] * z_emb]
            embeds.append(emb)
        return np.array(embeds)

    def emb2xi(self, embed, tt=None):
        # embed is a vector of size 3*size, where the first size elements correspond to x-axis, the next size to y-axis, and the last size to z-axis
        if tt is None:
            # Use cached bases
            if self.bases is None:
                raise ValueError("Bases not precomputed. Call set_spacing(tt) first.")
            return self.bases.transpose(1, 2, 0) @ embed

        x_xi = self.x_poly_fn(tt)[:, :self.size]
        y_xi = self.y_poly_fn(tt)[:, :self.size]
        z_xi = self.z_poly_fn(tt)[:, :self.size]

        x_coef = embed[:self.size]
        y_coef = embed[self.size:2 * self.size]
        z_coef = embed[2 * self.size:3 * self.size]
        xi = np.c_[
            np.zeros(len(tt)),
            np.zeros(len(tt)),
            np.zeros(len(tt)),
            x_xi @ x_coef,
            y_xi @ y_coef,
            z_xi @ z_coef,
        ]
        return xi

    def metric(self, xx):
        xi = self.emb2xi(xx)
        gamma = npoly3d.shape_exp(xi)
        bases = self.bases
        return npoly3d.metric(gamma, self.gab, bases)


def load_bases(gab_diag):
    # gab_diag is a list of 6 elements, where 1 means the corresponding dimension is included in the inner product
    gab = np.diag(gab_diag)

    # [1,1,0] -> '', [1,1,1] -> '1', [0,0,1] -> '2'
    with open(f'bases_fns/_orthopoly.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp0 = [sym.sympify(ps) for ps in polys_string]
    coeff0_fn = load_coeffs('bases_fns/_polycoeffs.yaml')

    with open(f'bases_fns/_orthopoly1.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp1 = [sym.sympify(ps) for ps in polys_string]
    coeff1_fn = load_coeffs('bases_fns/_polycoeffs1.yaml')

    with open(f'bases_fns/_orthopoly2.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp2 = [sym.sympify(ps) for ps in polys_string]
    coeff2_fn = load_coeffs('bases_fns/_polycoeffs2.yaml')

    x_fn = None
    y_fn = None
    z_fn = None
