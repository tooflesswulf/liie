import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import root
from tqdm.auto import tqdm

from .symbols import geodesic_equation


def compute_geodesic(x0, v0, metric, length=1.0, num_points=100, eps=1e-5, progress=False):
    """
    Integrate the geodesic ODE from x0 in direction v0 for the given arc length.

    v0 is normalised to unit speed under metric(x0) before integration.

    Returns
    -------
    path : (num_points, dim)
    """
    dim = len(x0)
    g_start = metric(x0)
    v0 = np.array(v0, dtype=float)
    norm = np.sqrt(v0 @ g_start @ v0)
    if norm > 0:
        v0 = v0 / norm
    y0 = np.r_[x0, v0]

    def eq(t, y):
        return geodesic_equation(y, metric, eps=eps)

    if progress:
        pbar = tqdm(total=100, desc='compute_geodesic')
        last_t = [0.0]
        _eq = eq
        def eq(t, y):
            inc = int((t - last_t[0]) / length * 100)
            if inc > 0:
                pbar.update(inc)
                last_t[0] = t
            return _eq(t, y)

    sol = solve_ivp(eq, [0, length], y0,
                    t_eval=np.linspace(0, length, num_points),
                    method='RK45', rtol=1e-6, atol=1e-8)

    if progress:
        pbar.close()

    return sol.y[:dim, :].T


def _path_length(path, metric):
    """Arc length of a discrete path under metric: Σ_i √(Δx_i^T g(x_i) Δx_i)."""
    total = 0.0
    for i in range(len(path) - 1):
        dx = path[i + 1] - path[i]
        total += float(np.sqrt(dx @ metric(path[i]) @ dx))
    return total


def dist_geo(x1, x2, metric, path=None, eps=1e-5, tol=1e-6, progress=False,
             return_path=False, num_points=100):
    """
    Geodesic distance between x1 and x2.

    If ``path`` is provided (e.g. from ``geodesic_bvp``), the arc length is
    computed directly along that path without any shooting:

        L = Σ_i √(Δx_i^T g(x_i) Δx_i)

    Otherwise, a shooting method is used: the initial velocity v0 at x1 that
    makes the geodesic reach x2 at t=1 is found via root-finding, and the
    distance is √(g(x1)(v0, v0)).

    Parameters
    ----------
    path : (N, dim) array, optional
        Pre-computed geodesic path, e.g. from ``geodesic_bvp``.
    return_path : bool
        Only used when ``path`` is None. If True, return (dist, path).
    """
    if path is not None:
        return _path_length(np.asarray(path), metric)

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    dim = len(x1)

    if progress:
        shoot_pbar = tqdm(total=100, desc='dist_geo shooting')

    def shoot(v0):
        if progress:
            shoot_pbar.reset()
            last_t = [0.0]
        def eq(t, y):
            if progress:
                inc = int((t - last_t[0]) * 100)
                if inc > 0:
                    shoot_pbar.update(inc)
                    last_t[0] = t
            return geodesic_equation(y, metric, eps=eps)
        sol = solve_ivp(eq, [0.0, 1.0], np.r_[x1, v0],
                        method='RK45', rtol=1e-8, atol=1e-10)
        return sol.y[:dim, -1] - x2

    sol = root(shoot, x2 - x1, tol=tol)

    if progress:
        shoot_pbar.close()

    if not sol.success:
        raise RuntimeError(f"Geodesic shooting failed: {sol.message}")

    v0 = sol.x
    dist = float(np.sqrt(v0 @ metric(x1) @ v0))

    if return_path:
        computed_path = compute_geodesic(x1, v0, metric, length=1.0,
                                         num_points=num_points, eps=eps, progress=progress)
        return dist, computed_path

    return dist


def geodesic_bvp(x1, x2, metric, num_points=100, eps=1e-5, tol=1e-3, progress=False):
    """
    Compute the geodesic path between x1 and x2 as a two-point BVP.

    Solves:  ẋ = v,  v̇ = -Γ^k_{ij} v^i v^j
    subject to x(0) = x1, x(1) = x2.
    Initial guess: straight line with constant velocity.

    Returns
    -------
    path : (num_points, dim)
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    dim = len(x1)

    def fun(t, y):
        return np.array([geodesic_equation(y[:, i], metric, eps=eps)
                         for i in range(y.shape[1])]).T

    def bc(ya, yb):
        return np.r_[ya[:dim] - x1, yb[:dim] - x2]

    t_init = np.linspace(0, 1, num_points)
    y_init = np.zeros((2 * dim, num_points))
    y_init[:dim] = x1[:, None] + np.outer(x2 - x1, t_init)
    y_init[dim:] = (x2 - x1)[:, None]

    if progress:
        pbar = tqdm(desc='geodesic_bvp', unit=' evals')
        _fun = fun
        def fun(t, y):
            dydt = _fun(t, y)
            dy_fd = np.gradient(y, t, axis=1)
            res = np.linalg.norm(dy_fd - dydt) / y.shape[1]
            pbar.update(1)
            pbar.set_postfix({'ode_res': f'{res:.2e}'})
            return dydt

    sol = solve_bvp(fun, bc, t_init, y_init, tol=tol)

    if progress:
        pbar.close()

    if not sol.success:
        raise RuntimeError(f"BVP geodesic failed: {sol.message}")

    return sol.sol(np.linspace(0, 1, num_points))[:dim].T


def geodesic_bvp_continuation(x1, x2, metric, num_points=100, eps=1e-5, tol=1e-3,
                               n_steps=10, progress=False):
    """
    Compute the geodesic between x1 and x2 via homotopy continuation.

    Solves a sequence of BVPs with interpolated metrics:
        metric_s(x) = (1 - s) * I  +  s * metric(x),   s ∈ [0, 1]

    At s=0 the metric is flat and the straight-line path is the exact solution.
    Each step warm-starts from the previous solution. Step size halves on failure.

    Returns
    -------
    path : (num_points, dim)
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    dim = len(x1)

    def bc(ya, yb):
        return np.r_[ya[:dim] - x1, yb[:dim] - x2]

    def make_fun(s, pbar=None):
        def fun(t, y):
            result = np.zeros_like(y)
            for i in range(y.shape[1]):
                xi = y[:dim, i]
                metric_s = (1 - s) * np.eye(dim) + s * metric(xi)
                result[:, i] = geodesic_equation(y[:, i], lambda x, m=metric_s: m, eps=eps)
            if pbar is not None:
                pbar.set_postfix({'s': f'{s:.2f}', 'n_pts': y.shape[1]})
                pbar.update(1)
            return result
        return fun

    # s=0: straight line is the exact solution
    t_cur = np.linspace(0, 1, num_points)
    y_cur = np.zeros((2 * dim, num_points))
    y_cur[:dim] = x1[:, None] + np.outer(x2 - x1, t_cur)
    y_cur[dim:] = (x2 - x1)[:, None]

    pbar = tqdm(desc='continuation', unit=' evals') if progress else None

    s, ds = 0.0, 1.0 / n_steps
    while s < 1.0 - 1e-12:
        s_next = min(s + ds, 1.0)
        sol = solve_bvp(make_fun(s_next, pbar), bc, t_cur, y_cur, tol=tol)
        if sol.success:
            t_cur, y_cur = sol.x, sol.y
            s = s_next
        else:
            ds /= 2
            if ds < 1e-6:
                raise RuntimeError(f"Continuation failed at s={s:.4f}: step too small")

    if pbar is not None:
        pbar.close()

    return sol.sol(np.linspace(0, 1, num_points))[:dim].T
