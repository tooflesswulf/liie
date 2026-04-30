import numpy as np
import pinocchio as pin

import matplotlib.pyplot as plt
import yaml
import sympy as sym


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
    integrand = np.einsum('si,ij,sj->s', gdot1, gab, gdot2)
    return np.trapezoid(integrand, dx=1 / (N - 1))


def shape_exp(xi):
    """Integrate a twist trajectory xi(s) to get the SE(3) trajectory gamma(s).
    Adds a baseline unit translation along z so the arm has unit arc-length."""
    N = xi.shape[0]
    xi = xi + np.array([0, 0, 1, 0, 0, 0])  # forward motion along z
    gi = np.eye(4)
    gamma = [gi]
    ds = 1 / (N - 1)
    for i in range(1, N):
        accu = np.array(pin.exp(xi[i - 1] * ds))
        gi = gi @ accu
        gamma.append(gi)
    return np.array(gamma)


if __name__ == '__main__':
    N = 100

    _gamma = np.zeros((N, 6))
    _gamma[:, 2] = np.linspace(0, 1, N)
    gamma = np.array([np.array(pin.exp(gi)) for gi in _gamma])

    sx = sym.symbols('t')
    sfx = ''  
    gab = np.diag([1, 1, 1, 0, 0, 0])

    with open(f'_orthopoly{sfx}.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp = [sym.sympify(ps) for ps in polys_string]
    print('loaded bases')

    def eval_poly(x):
        return np.array([pi.subs({sx: x}) for pi in pp]).astype(float)

    tt = np.linspace(0, 1, N)
    npoly = np.array([eval_poly(ti) for ti in tt])

    # Use first two basis polys as the profile for two bending directions:
    # xi bends about the x-axis, xi2 bends about the y-axis
    xi = np.zeros((N, 6))
    xi[:, 3] = npoly[:, 0]   # omega_x profile
    xi2 = np.zeros((N, 6))
    xi2[:, 4] = npoly[:, 1]  # omega_y profile

    print("Inner products at straight reference configuration:")
    print(f"  <xi,  xi>  = {inner_product(gamma, gab, xi,  xi):.6f}")
    print(f"  <xi,  xi2> = {inner_product(gamma, gab, xi,  xi2):.6f}")
    print(f"  <xi2, xi2> = {inner_product(gamma, gab, xi2, xi2):.6f}")

    # Build a bent configuration with shape_exp, then evaluate the Gram matrix there
    gamma_bent = shape_exp(xi / 2 + xi2 / 2)
    g00 = inner_product(gamma_bent, gab, xi,  xi)
    g01 = inner_product(gamma_bent, gab, xi,  xi2)
    g11 = inner_product(gamma_bent, gab, xi2, xi2)
    print("\nGram matrix at bent configuration:")
    print(np.array([[g00, g01],
                    [g01, g11]]))

    # Plot reference and bent arm
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(gamma[:, 0, 3], gamma[:, 1, 3], gamma[:, 2, 3],
            '--', alpha=0.5, label='straight reference')
    ax.plot(gamma_bent[:, 0, 3], gamma_bent[:, 1, 3], gamma_bent[:, 2, 3],
            label='bent (xi/2 + xi2/2)')

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        ax.set_box_aspect([1, 1, 1])
    ax.legend()
    ax.set_title("3D continuum arm from shape_exp")
    plt.show()