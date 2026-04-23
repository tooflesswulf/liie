import numpy as np
import pinocchio as pin
from scipy import integrate

import matplotlib.pyplot as plt


def gammadot(gamma, xi):
    # gammadot(s) = Ad(gamma(s)).inv() @ int_0^s Ad(gamma(t)) @ xi dt
    N = gamma.shape[0]

    Ad_gamma_t = np.array([pin.SE3(gamma[ti]).action for ti in range(N)])
    Ad_gammainv_t = np.array([pin.SE3(gamma[ti]).inverse().action for ti in range(N)])
    return _gammadot_help(Ad_gamma_t, Ad_gammainv_t, xi)


def _gammadot_help(AdG, AdGinv, xi):
    integrand = AdG @ xi[:, :, None]
    cum_integral = integrate.cumulative_trapezoid(integrand, dx=1 / (AdG.shape[0] - 1), axis=0, initial=0)
    return (AdGinv @ cum_integral)[:, :, 0]


def inner_product(gamma, gab, xi1, xi2):
    N = gamma.shape[0]
    gdot1 = gammadot(gamma, xi1)
    gdot2 = gammadot(gamma, xi2)

    # Compute the inner product as an integral over s
    integrand = np.einsum('si,ij,sj->s', gdot1, gab, gdot2)
    return np.trapezoid(integrand, dx=1 / (N - 1))


def metric(gamma, gab, xi_bases):
    N = gamma.shape[0]

    Ad_gamma = np.array([pin.SE3(gamma[ti]).action for ti in range(N)])
    Ad_gammainv = np.array([pin.SE3(gamma[ti]).inverse().action for ti in range(N)])
    gdot_bases = np.array([_gammadot_help(Ad_gamma, Ad_gammainv, xi) for xi in xi_bases])

    trapezoid_dot = np.ones(N) / (N - 1)
    trapezoid_dot[0] /= 2
    trapezoid_dot[-1] /= 2

    mm = np.einsum('nta,t,ab,mtb->nm', gdot_bases, trapezoid_dot, gab, gdot_bases, optimize=True)
    return mm


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


def skew(a):
    """3×3 skew-symmetric matrix for vector a."""
    return np.array([
        [ 0,    -a[2],  a[1]],
        [ a[2],  0,    -a[0]],
        [-a[1],  a[0],  0   ],
    ])


def ad6(xi):
    """
    6×6 adjoint (small ad) matrix for xi = [v, omega] in pinocchio convention
    (linear velocity first, then angular).

    ad(xi) = [[skew(omega), skew(v)],
              [0,           skew(omega)]]

    Satisfies: ad(xi) @ eta = [xi, eta]  (Lie bracket on se(3)).
    """
    v, omega = xi[:3], xi[3:]
    return np.block([
        [skew(omega), skew(v)      ],
        [np.zeros((3, 3)), skew(omega)],
    ])


def metric_jacobian(gamma, gab, xi_bases):
    """
    Compute the partial derivatives of the Riemannian metric with respect to
    the embedding coordinates.

    Parameters
    ----------
    gamma : (N, 4, 4) array
        SE(3) path produced by shape_exp.
    gab : (6, 6) array
        Inner-product matrix on se(3) fibres (e.g. diag([1,1,1,0,0,1])).
    xi_bases : (n, N, 6) array
        Twist-field basis vectors evaluated on the N-point grid.

    Returns
    -------
    dg : (n, n, n) array
        dg[k, i, j]  =  ∂g_{ij} / ∂xx_k
    """
    N = gamma.shape[0]
    n = xi_bases.shape[0]
    dt = 1.0 / (N - 1)

    # Build Ad(gamma) and Ad(gamma)^{-1} once
    AdG    = np.array([pin.SE3(gamma[t]).action         for t in range(N)])  # (N, 6, 6)
    AdGinv = np.array([pin.SE3(gamma[t]).inverse().action for t in range(N)])  # (N, 6, 6)

    # Precompute γ-dots for every basis: gdots[k, t, :] = η_k(t)
    # η_k = Ad(γ)^{-1} ∫_0^s Ad(γ) b_k dt
    gdots = np.array([_gammadot_help(AdG, AdGinv, xi_bases[k]) for k in range(n)])
    # gdots : (n, N, 6)

    # ad(η_k(t)) matrices — shape (n, N, 6, 6)
    ad_mats = np.array([[ad6(gdots[k, t]) for t in range(N)] for k in range(n)])

    # Trapezoid weights
    w = np.ones(N) * dt
    w[0]  /= 2
    w[-1] /= 2

    # ---------------------------------------------------------------
    # Correct variational formula:
    #   ∂_k γ̇_i(s) = Ad(γ(s))⁻¹ Ψ_{ki}(s)  −  ad(η_k(s)) γ̇_i(s)
    #
    # where  Ψ_{ki}(s) = ∫_0^s Ad(γ(u)) ad(η_k(u)) b_i(u) du
    #
    # Note: the subtracted term is  −ad(η_k) γ̇_i, NOT Ad(γ)⁻¹ ad(η_k) Ad(γ) γ̇_i.
    # Those differ because Ad(γ)⁻¹ and ad(η_k) don't generally commute.
    # ---------------------------------------------------------------

    # Ψ_{ki}(s) = ∫_0^s Ad(γ(u)) ad(η_k(u)) b_i(u) du
    # Ad(γ) ad(η_k) : (n, N, 6, 6)
    AdG_ad = np.einsum('tab,ktbc->ktac', AdG, ad_mats, optimize=True)  # (n, N, 6, 6)

    # integrand[k, i, t, :] = Ad(γ(t)) ad(η_k(t)) b_i(t)
    integrand_Psi = np.einsum('ktac,itc->kita', AdG_ad, xi_bases, optimize=True)  # (n, n, N, 6)

    # Cumulative trapezoid over time axis (axis=2)
    Psi = integrate.cumulative_trapezoid(integrand_Psi, dx=dt, axis=2, initial=0)
    # Psi : (n, n, N, 6)

    # Term 1: Ad(γ(s))⁻¹ Ψ_{ki}(s)
    term1 = np.einsum('tab,kitb->kita', AdGinv, Psi, optimize=True)       # (n, n, N, 6)

    # Term 2: −ad(η_k(s)) γ̇_i(s)
    term2 = -np.einsum('ktab,itb->kita', ad_mats, gdots, optimize=True)   # (n, n, N, 6)

    d_gd = term1 + term2  # (n, n, N, 6)
    # ---------------------------------------------------------------
    # ∂g_{ij}/∂xx_k = ∫ (∂_k γ̇_i)^T G γ̇_j  +  γ̇_i^T G (∂_k γ̇_j)  ds
    # ---------------------------------------------------------------
    T1 = np.einsum('kita,ab,jtb,t->kij', d_gd, gab, gdots, w)
    T2 = np.einsum('ita,ab,kjtb,t->kij', gdots, gab, d_gd, w)

    return T1 + T2


if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # Finite-difference check for metric_jacobian
    # -----------------------------------------------------------------------
    import sys

    N = 60   # grid points along arm
    tt = np.linspace(0, 1, N)

    # Small set of basis functions: polynomials in the angular (omega) channels
    # xi_bases[k, s, c]  —  channel layout: [v_x, v_y, v_z, omega_x, omega_y, omega_z]
    n_bases = 5
    xi_bases = np.zeros((n_bases, N, 6))
    for k in range(n_bases):
        xi_bases[k, :, 3 + (k % 3)] = tt ** k   # ω_x, ω_y, ω_z, ω_x, ω_y channels

    gab = np.diag([1., 1., 1., 0., 0., 1.])     # standard arm inner product

    # Random (small) embedding coordinates so gamma stays near identity
    rng = np.random.default_rng(0)
    xx = rng.normal(scale=0.05, size=n_bases)

    # Helper: build gamma from embedding coords
    def gamma_from_xx(c):
        xi = np.einsum('k,kts->ts', c, xi_bases)
        return shape_exp(xi)

    gamma = gamma_from_xx(xx)

    # --- Analytic jacobian ---------------------------------------------------
    dg_analytic = metric_jacobian(gamma, gab, xi_bases)

    # --- Finite-difference jacobian ------------------------------------------
    eps = 1e-5
    dg_fd = np.zeros((n_bases, n_bases, n_bases))
    for k in range(n_bases):
        xx_p, xx_m = xx.copy(), xx.copy()
        xx_p[k] += eps
        xx_m[k] -= eps
        g_p = metric(gamma_from_xx(xx_p), gab, xi_bases)
        g_m = metric(gamma_from_xx(xx_m), gab, xi_bases)
        dg_fd[k] = (g_p - g_m) / (2 * eps)

    # --- Report --------------------------------------------------------------
    err     = np.abs(dg_analytic - dg_fd)
    rel_err = err / (np.abs(dg_fd) + 1e-12)

    print("metric_jacobian finite-difference check")
    print("=" * 45)
    print(f"  n_bases = {n_bases},  N = {N},  eps = {eps:.0e}")
    print(f"  Max abs error   : {err.max():.3e}")
    print(f"  Mean abs error  : {err.mean():.3e}")
    print(f"  Max rel error   : {rel_err.max():.3e}")
    print()
    print("Analytic dg[k,i,j]:")
    for k in range(n_bases):
        print(f"  k={k}: {dg_analytic[k].round(6)}")
    print()
    print("FD      dg[k,i,j]:")
    for k in range(n_bases):
        print(f"  k={k}: {dg_fd[k].round(6)}")

    ok = err.max() < 1e-6
    print()
    print("PASS" if ok else "FAIL", f"(threshold 1e-6)")
    sys.exit(0 if ok else 1)
