import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from .symbols import compute_christoffel_symbols, compute_christoffel_symbols_analytic


def levi_civita_connection(x, xdot, v, metric, eps=1e-5, metric_jacobian=None):
    """
    Covariant derivative of vector v in direction xdot at point x under the
    Levi-Civita connection:

        (∇_{ẋ} v)^k = Γ^k_{ij}(x) ẋ^i v^j

    Parameters
    ----------
    x : (dim,) array
        Base point.
    xdot : (dim,) array
        Tangent direction (the "velocity" of the curve).
    v : (dim,) array
        Vector to differentiate.
    metric : callable
        x -> (dim, dim) Riemannian metric tensor.
    eps : float
        Finite-difference step for Christoffel symbol computation.
        Ignored when metric_jacobian is supplied.
    metric_jacobian : callable, optional
        x -> (dim, dim, dim) analytic metric Jacobian,
        dg[k, i, j] = ∂g_{ij}/∂x^k.
        When provided, Christoffel symbols are computed analytically.

    Returns
    -------
    nabla_v : (dim,) array
        The connection 1-form applied to (xdot, v):  Γ^k_{ij} ẋ^i v^j.
    """
    if metric_jacobian is not None:
        Gamma = compute_christoffel_symbols_analytic(x, metric, metric_jacobian)
    else:
        Gamma = compute_christoffel_symbols(x, metric, eps=eps)

    return np.einsum('kij,i,j->k', Gamma, xdot, v)


def parallel_transport(v0, path, metric, eps=1e-5, metric_jacobian=None,
                       return_all=False, rtol=1e-6, atol=1e-8):
    """
    Parallel transport a vector along a discrete path under the Levi-Civita
    connection.

    The discrete path is first fitted with a cubic spline to produce a smooth
    curve γ(t), t ∈ [0, 1].  The parallel transport ODE

        dV^k/dt = -Γ^k_{ij}(γ(t)) γ̇^i(t) V^j(t)

    is then integrated with an adaptive RK45 solver.

    Parameters
    ----------
    v0 : (dim,) array
        Initial vector at path[0].
    path : (N, dim) array
        Discrete curve along which to transport v0.  N ≥ 2.
    metric : callable
        x -> (dim, dim) Riemannian metric tensor.
    eps : float
        Finite-difference step for Christoffel symbols (FD fallback).
    metric_jacobian : callable, optional
        x -> (dim, dim, dim) analytic metric Jacobian,
        dg[k, i, j] = ∂g_{ij}/∂x^k.
        When provided, Christoffel symbols are computed analytically
        via compute_christoffel_symbols_analytic rather than FD.
    return_all : bool
        If False (default), return the transported vector at path[-1].
        If True, return transported vectors at every node of ``path``.
    rtol, atol : float
        Relative / absolute tolerances for the ODE solver.

    Returns
    -------
    v : (dim,) array   if return_all is False
        Transported vector at the end of the path.
    v : (N, dim) array  if return_all is True
        Transported vector at each path node.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError(f"path must be 2-D, got shape {path.shape}")
    N, dim = path.shape
    if N < 2:
        raise ValueError("path must have at least 2 points")

    t_nodes = np.linspace(0.0, 1.0, N)

    # Fit a cubic spline so we get smooth position and velocity
    cs     = CubicSpline(t_nodes, path)
    cs_dot = cs.derivative()

    if metric_jacobian is not None:
        def _christoffel(x):
            return compute_christoffel_symbols_analytic(x, metric, metric_jacobian)
    else:
        def _christoffel(x):
            return compute_christoffel_symbols(x, metric, eps=eps)

    def ode(t, v):
        x    = cs(t)
        xdot = cs_dot(t)
        Gamma = _christoffel(x)
        return -np.einsum('kij,i,j->k', Gamma, xdot, v)

    t_eval = t_nodes if return_all else None
    sol = solve_ivp(ode, [0.0, 1.0], np.asarray(v0, dtype=float),
                    t_eval=t_eval, method='RK45', rtol=rtol, atol=atol)

    if not sol.success:
        raise RuntimeError(f"Parallel transport integration failed: {sol.message}")

    return sol.y.T if return_all else sol.y[:, -1]
