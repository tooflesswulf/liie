import numpy as np
import scipy.special
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import root
from tqdm.auto import tqdm
import yaml
import sympy as sym
import matplotlib.pyplot as plt


def compute_christoffel_symbols(x, metric, eps=1e-5):
    """
    Compute Christoffel symbols of the second kind via finite differences.
    Γ^k_{ij} = (1/2) g^{kl} (∂g_{jl}/∂x^i + ∂g_{il}/∂x^j - ∂g_{ij}/∂x^l)
    """
    dim = len(x)
    g = metric(x)
    g_inv = np.linalg.inv(g)

    # Compute metric derivatives via finite differences
    dg = []
    for i in range(dim):
        dx = np.zeros(dim)
        dx[i] = eps
        dg.append((metric(x + dx) - metric(x - dx)) / (2 * eps))

    Gamma = np.zeros((dim, dim, dim))  # Gamma[k, i, j]

    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                christoffel_sum = 0.0
                for l in range(dim):
                    christoffel_sum += g_inv[k, l] * (
                        dg[i][j, l] +
                        dg[j][i, l] -
                        dg[l][i, j]
                    )
                Gamma[k, i, j] = 0.5 * christoffel_sum

    # Ensure symmetry in lower indices: Γ^k_{ij} = Γ^k_{ji}
    # Shouldn't be necessary but just in case of numerical issues
    for k in range(dim):
        for i in range(dim):
            for j in range(i + 1, dim):
                avg = (Gamma[k, i, j] + Gamma[k, j, i]) / 2
                Gamma[k, i, j] = avg
                Gamma[k, j, i] = avg

    return Gamma


def compute_ricci_curvature(x, metric, eps=1e-5):
    """
    Compute the Ricci curvature at a point using the Riemann curvature tensor.

    For a 2D surface, Ricci curvature is given by:
    R = g^{ij} R_{ij}

    where R_{ij} is the Ricci tensor and g^{ij} is the inverse metric tensor.

    In terms of Christoffel symbols and their derivatives:
    R^i_{jkl} = ∂Γ^i_{jl}/∂x^k - ∂Γ^i_{jk}/∂x^l + Γ^m_{jl}Γ^i_{mk} - Γ^m_{jk}Γ^i_{ml}

    Parameters:
    -----------
    c1, c2 : float
        Coordinates at which to compute curvature
    bases : array
        Basis functions for metric computation
    eps : float
        Step size for finite differences

    Returns:
    --------
    K : float
        Gaussian curvature at the point
    """
    dim = len(x)
    g = metric(x)
    ginv = np.linalg.pinv(g)
    det_g = np.linalg.det(g)

    Gamma = compute_christoffel_symbols(x, metric, eps)
    dGamma = []
    for i in range(dim):
        dx = np.zeros(dim)
        dx[i] = eps
        dx_plus = compute_christoffel_symbols(x + dx, metric, eps)
        dx_minus = compute_christoffel_symbols(x - dx, metric, eps)
        dGamma.append((dx_plus - dx_minus) / (2 * eps))

    accu = 0.0
    for mu in range(dim):
        for nu in range(dim):
            for l in range(dim):
                accu += g[mu, nu] * (dGamma[l][l, mu, nu] - dGamma[nu][l, mu, l])

    # Ricci tensor
    R_ij = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            accu1 = 0
            accu2 = 0
            for k in range(dim):
                accu1 += dGamma[k][k, i, j] - dGamma[j][k, i, k]
                for m in range(dim):
                    accu2 += Gamma[k, i, j] * Gamma[m, k, m] - Gamma[k, i, m] * Gamma[m, j, k]
            R_ij[i, j] = accu1 + accu2

    # Ricci scalar
    R = np.sum(ginv * R_ij)
    return R


def montecarlo_integration_samples(domain, num_samples=1000):
    """
    Sample points uniformly from a given domain for Monte Carlo integration.

    Parameters:
    -----------
    domain : dict
        Domain specification with keys 'type' and parameters:
        - type='mesh': simplex mesh with keys 'points' (N*dim), 'simplices' (M*(dim+1))
            Samples uniformly from the interior of mesh simplices.
    num_samples : int
        Number of random sample points to generate.

    Returns:
    --------
    samples : array
        Array of shape (num_samples, dim) containing sampled points.
    """
    if domain['type'] != 'mesh':
        raise ValueError(f"Only mesh domains are supported. Got: {domain['type']}")

    points = np.asarray(domain['points'])
    simplices = np.asarray(domain['simplices'])
    dim = points.shape[1]

    # Compute volume of each simplex
    simplex_volumes = []
    for simplex in simplices:
        vertex_coords = points[simplex]
        v0 = vertex_coords[0]
        edge_matrix = vertex_coords[1:] - v0  # (dim) × (dim) matrix
        vol = np.abs(np.linalg.det(edge_matrix)) / scipy.special.factorial(dim)
        simplex_volumes.append(vol)

    simplex_volumes = np.array(simplex_volumes)
    total_volume = np.sum(simplex_volumes)

    # Sample simplices proportionally to their volumes
    simplex_weights = simplex_volumes / total_volume
    sampled_simplices = np.random.choice(len(simplices), size=num_samples, p=simplex_weights)

    # Generate samples uniformly within each selected simplex using barycentric coordinates
    samples = np.zeros((num_samples, dim))
    for idx in range(num_samples):
        simplex_idx = sampled_simplices[idx]
        simplex = simplices[simplex_idx]
        vertex_coords = points[simplex]

        # Random barycentric coordinates
        bary = np.concatenate([[0], np.sort(np.random.uniform(0, 1, dim)), [1]])
        bary = np.diff(bary)

        # Weighted sum of vertices
        samples[idx] = np.sum(vertex_coords * bary[:, np.newaxis], axis=0)

    return samples, total_volume

def integrate_ricci_scalar(metric, domain, num_samples=1000, eps=1e-5, progress=False):
    """
    Integrate the Ricci scalar over a given domain in n-dimensional space using Monte Carlo integration.

    ∫_V R √det(g) d^n x ≈ V * <R √det(g)>

    where V is the volume of the domain and the average is over random samples.

    Parameters:
    -----------
    metric : callable
        Metric function metric(x) that returns a (dim, dim) matrix.
    domain : dict
        Domain specification with keys 'type' and parameters:
        - type='mesh': simplex mesh with keys 'points' (N*dim), 'simplices' (M*(dim+1))
            Samples uniformly from the interior of mesh simplices.
    num_samples : int
        Number of random sample points for Monte Carlo integration.
    eps : float
        Step size for finite differences in curvature computation.

    Returns:
    --------
    integral : float
        Approximate value of ∫_V R √det(g) d^n x
    error_estimate : float
        Standard error estimate from Monte Carlo sampling
    """
    samples, volume = montecarlo_integration_samples(domain, num_samples=num_samples)

    # Evaluate R * √det(g) at each sample
    integrand_values = []
    iterator = tqdm(samples, desc='Ricci MC', leave=False) if progress else samples
    for sample in iterator:
        try:
            R = compute_ricci_curvature(sample, metric, eps)
            g = metric(sample)
            det_g = np.linalg.det(g)
            sqrt_det_g = np.sqrt(np.abs(det_g))  # Use abs for robustness
            integrand_values.append(R * sqrt_det_g)
        except:
            # Skip samples where metric evaluation fails (e.g., outside domain)
            continue

    if len(integrand_values) == 0:
        raise ValueError("No valid samples in domain. Check bounds/radius and metric definition.")

    integrand_values = np.array(integrand_values)

    # Monte Carlo estimate: I ≈ V * <f>
    mean_integrand = np.mean(integrand_values)
    std_integrand = np.std(integrand_values)

    integral = volume * mean_integrand
    error_estimate = volume * std_integrand / np.sqrt(len(integrand_values))

    return integral, error_estimate


def _orient_ccw_2d(points, simplices):
    """Return triangles re-ordered so each is CCW in coordinates."""
    out = np.asarray(simplices).copy()
    for k, tri in enumerate(out):
        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if cross < 0:
            out[k] = [tri[0], tri[2], tri[1]]
    return out


def _boundary_cycles_2d(simplices):
    """
    Extract ordered boundary cycles from a CCW-oriented 2D triangle mesh.
    Returns a list of cycles; each cycle is a list of vertex indices traversed
    in CCW order along the boundary (interior on the left).
    """
    edge_uses = {}
    for tri in simplices:
        for k in range(3):
            i, j = int(tri[k]), int(tri[(k + 1) % 3])
            edge_uses[(i, j)] = edge_uses.get((i, j), 0) + 1

    # A directed edge is on the boundary iff its reverse is not also present.
    next_v = {}
    for (i, j) in edge_uses:
        if (j, i) not in edge_uses:
            next_v[i] = j

    cycles = []
    visited = set()
    for start in list(next_v.keys()):
        if start in visited:
            continue
        cycle = [start]
        visited.add(start)
        cur = next_v.get(start)
        while cur is not None and cur != start:
            if cur in visited:
                break
            cycle.append(cur)
            visited.add(cur)
            cur = next_v.get(cur)
        cycles.append(cycle)
    return cycles


def _J_rot_2d(g, v):
    """+90° rotation of v in the metric g (oriented). J(1,0)=(0,1) when g=I."""
    det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
    sd = np.sqrt(det)
    return np.array([
        -(g[0, 1] * v[0] + g[1, 1] * v[1]) / sd,
         (g[0, 0] * v[0] + g[0, 1] * v[1]) / sd,
    ])


def integrate_boundary_term_2d(metric, domain, n_quad=16, eps=1e-5, progress=False):
    """
    Compute the boundary contribution to Gauss-Bonnet on a 2D triangulated
    domain with piecewise-linear boundary:

        B(∂M) = ∮_∂M k_g ds  +  Σ_corners θ_i

    where θ_i are signed exterior (turning) angles in the metric.

    Returns
    -------
    kg_integral : float
        ∮ k_g ds along the straight (in coordinates) boundary segments.
    corner_sum : float
        Σ exterior angles at boundary vertices.
    """
    points = np.asarray(domain['points'], dtype=float)
    simplices = _orient_ccw_2d(points, domain['simplices'])
    cycles = _boundary_cycles_2d(simplices)

    # Gauss-Legendre quadrature on [0, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    kg_integral = 0.0
    corner_sum = 0.0

    total_segments = sum(len(c) for c in cycles if len(c) >= 2)
    seg_bar = tqdm(total=total_segments, desc='Boundary k_g', leave=False) if progress else None

    for cycle in cycles:
        N = len(cycle)
        if N < 2:
            continue

        # ∮ k_g ds : integrate along each straight (in coordinates) segment.
        for k in range(N):
            i = cycle[k]
            j = cycle[(k + 1) % N]
            a = points[i]
            b = points[j]
            v = b - a
            seg = 0.0
            for t, w in zip(nodes, weights):
                p = a + t * v
                g = metric(p)
                Gamma = compute_christoffel_symbols(p, metric, eps=eps)
                # Coordinate acceleration of the straight line: A^k = Γ^k_{ij} v^i v^j
                A = np.einsum('kij,i,j->k', Gamma, v, v)
                Jv = _J_rot_2d(g, v)
                s2 = float(v @ g @ v)
                # k_g ds = g(A, Jv) / s^2 dt   (reparameterization-invariant)
                seg += w * float(A @ g @ Jv) / s2
            kg_integral += seg
            if seg_bar is not None:
                seg_bar.update(1)

        # Σ exterior angles at corners
        for k in range(N):
            i_prev = cycle[(k - 1) % N]
            i_curr = cycle[k]
            i_next = cycle[(k + 1) % N]
            p = points[i_curr]
            T_in = points[i_curr] - points[i_prev]
            T_out = points[i_next] - points[i_curr]
            g = metric(p)
            T_in = T_in / np.sqrt(T_in @ g @ T_in)
            T_out = T_out / np.sqrt(T_out @ g @ T_out)
            cos_a = float(T_in @ g @ T_out)
            sin_a = float(_J_rot_2d(g, T_in) @ g @ T_out)
            corner_sum += np.arctan2(sin_a, cos_a)

    if seg_bar is not None:
        seg_bar.close()

    return kg_integral, corner_sum


def euler_characteristic(metric, domain, num_samples=5000, eps=1e-5,
                         include_boundary=True, n_quad=16, progress=True):
    """
    Compute the Euler characteristic of a 2D Riemannian manifold via the
    Gauss-Bonnet theorem (with boundary):

        ∫_M K dA + ∮_∂M k_g ds + Σ_corners θ_i = 2π χ(M)

    In 2D the Ricci scalar satisfies R = 2K, so

        χ(M) = ( (1/2) ∫_M R √det(g) d²x + ∮ k_g ds + Σ θ_i ) / (2π).

    The boundary of the mesh is treated as piecewise straight in coordinates,
    with corners at the vertices contributing exterior turning angles.

    Parameters
    ----------
    metric : callable
        metric(x) -> (2, 2) matrix.
    domain : dict
        Mesh domain as accepted by `integrate_ricci_scalar`.
    num_samples : int
        Monte Carlo sample count.
    eps : float
        Finite-difference step for curvature.

    Returns
    -------
    chi : float
        Estimated Euler characteristic (round to nearest integer for the
        topological invariant).
    error : float
        1-sigma Monte Carlo error on chi.
    """
    # Sanity-check dimension by probing the metric at a sample point.
    sample_pt = np.asarray(domain['points'])[0]
    g_probe = metric(sample_pt)
    dim = g_probe.shape[0]
    if dim != 2:
        raise NotImplementedError(
            "euler_characteristic is currently only implemented for 2D "
            "manifolds (Gauss-Bonnet). For dim > 2 the Chern-Gauss-Bonnet "
            "formula requires the Pfaffian of the Riemann tensor."
        )

    integral, error = integrate_ricci_scalar(
        metric, domain, num_samples=num_samples, eps=eps, progress=progress
    )
    interior_K = 0.5 * integral  # ∫ K dA = (1/2) ∫ R √g d²x in 2D

    boundary_kg = 0.0
    corner_sum = 0.0
    if include_boundary:
        boundary_kg, corner_sum = integrate_boundary_term_2d(
            metric, domain, n_quad=n_quad, eps=eps, progress=progress
        )

    chi = (interior_K + boundary_kg + corner_sum) / (2.0 * np.pi)
    return chi, 0.5 * error / (2.0 * np.pi)


def geodesic_equation(state, metric, eps=1e-5):
    """
    Geodesic equation in terms of Christoffel symbols:
    d^2 x^k / dt^2 + Γ^k_{ij} (dx^i/dt) (dx^j/dt) = 0

    State vector: [x^1, x^2, ..., x^n, dx^1/dt, dx^2/dt, ..., dx^n/dt]

    Returns the time derivative: [dx^i/dt, d^2x^i/dt^2]
    """

    dim = len(state) // 2
    x = state[:dim]
    xdot = state[dim:]
    Gamma = compute_christoffel_symbols(x, metric, eps=eps)

    # Acceleration Gamma[k, i, j] * xdot[i] * xdot[j]
    xddot = np.zeros(dim)
    for k in range(dim):
        xddot[k] = -np.sum(Gamma[k, :, :] * np.outer(xdot, xdot))

    return np.r_[xdot, xddot]


def compute_geodesic(x0, v0, metric, length=1.0, num_points=100, eps=1e-5, progress=False):
    """Compute a geodesic starting from x0 in the given direction."""
    # Normalize direction according to metric at start point
    dim = len(x0)
    g_start = metric(x0)
    v0 = np.array(v0)
    norm = np.sqrt(v0 @ g_start @ v0)
    if norm > 0:
        v0 = v0 / norm
    # Initial state: position + velocity
    y0 = np.r_[x0, v0]

    def eq(t, y):
        return geodesic_equation(y, metric)

    if progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=100, desc='compute_geodesic')
        last_t = [0.0]
        _eq = eq
        def eq(t, y):
            inc = int((t - last_t[0]) / length * 100)
            if inc > 0:
                pbar.update(inc)
                last_t[0] = t
            return _eq(t, y)

    # Integrate geodesic equation
    sol = solve_ivp(
        eq,
        [0, length],
        y0,
        t_eval=np.linspace(0, length, num_points),
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )

    if progress:
        pbar.close()

    return sol.y[:dim, :].T  # Return only positions


def dist_geo(x1, x2, metric, eps=1e-5, tol=1e-6, progress=False, return_path=False, num_points=100):
    """
    Compute the geodesic distance between two points x1 and x2.

    Uses a shooting method: find an initial velocity v0 at x1 such that the
    geodesic x(t) with x(0)=x1, ẋ(0)=v0 satisfies x(1)=x2. Since geodesics
    have constant speed under the metric, the distance is then √(g(x1)(v0, v0)).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    dim = len(x1)

    if progress:
        from tqdm.auto import tqdm
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
        y0 = np.r_[x1, v0]
        sol = solve_ivp(eq, [0.0, 1.0], y0, method='RK45', rtol=1e-8, atol=1e-10)
        return sol.y[:dim, -1] - x2

    # Initial guess: Euclidean displacement (correct in flat space)
    v0_guess = x2 - x1
    sol = root(shoot, v0_guess, tol=tol)

    if progress:
        shoot_pbar.close()

    if not sol.success:
        raise RuntimeError(f"Geodesic shooting failed to converge: {sol.message}")

    v0 = sol.x
    g0 = metric(x1)
    dist = float(np.sqrt(v0 @ g0 @ v0))

    if return_path:
        path = compute_geodesic(x1, v0, metric, length=1.0, num_points=num_points,
                                eps=eps, progress=progress)
        return dist, path

    return dist


def geodesic_bvp(x1, x2, metric, num_points=100, eps=1e-5, tol=1e-3, progress=False):
    """
    Compute the geodesic path between x1 and x2 by solving the geodesic BVP directly.

    Solves the geodesic ODE as a two-point boundary value problem:
        ẋ = v
        v̇ = -Γ^k_{ij} v^i v^j
    subject to x(0) = x1, x(1) = x2, using a straight-line initial guess.

    Returns path of shape (num_points, dim).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    dim = len(x1)

    def fun(t, y):
        # y: (2*dim, n) — evaluate geodesic equation column-wise
        return np.array([geodesic_equation(y[:, i], metric, eps=eps)
                         for i in range(y.shape[1])]).T

    def bc(ya, yb):
        return np.r_[ya[:dim] - x1, yb[:dim] - x2]

    # Initial guess: straight line with constant velocity
    t_init = np.linspace(0, 1, num_points)
    y_init = np.zeros((2 * dim, num_points))
    y_init[:dim] = x1[:, None] + np.outer(x2 - x1, t_init)
    y_init[dim:] = (x2 - x1)[:, None]

    if progress:
        pbar = tqdm(desc='geodesic_bvp', unit=' evals')
        _fun = fun
        def fun(t, y):
            dydt = _fun(t, y)
            # Collocation residual: how well does the ODE hold given current y?
            dy_fd = np.gradient(y, t, axis=1)
            res = np.linalg.norm(dy_fd - dydt) / y.shape[1]
            pbar.update(1)
            pbar.set_postfix({'ode_res': f'{res:.2e}'})
            return dydt

    sol = solve_bvp(fun, bc, t_init, y_init, tol=tol)

    if progress:
        pbar.close()

    if not sol.success:
        raise RuntimeError(f"BVP geodesic failed to converge: {sol.message}")

    t_out = np.linspace(0, 1, num_points)
    return sol.sol(t_out)[:dim].T  # (num_points, dim)


def geodesic_bvp_continuation(x1, x2, metric, num_points=100, eps=1e-5, tol=1e-3,
                               n_steps=10, progress=False):
    """
    Compute the geodesic between x1 and x2 via homotopy continuation.

    Solves a sequence of BVPs with interpolated metrics:
        metric_s(x) = (1 - s) * I  +  s * metric(x),   s in [0, 1]

    At s=0 the metric is flat and the straight-line path is the exact solution.
    Each step uses the previous solution as the warm start, so Newton stays
    well-conditioned throughout.

    Adaptive step size: if a BVP step fails, the step is halved and retried.

    Returns path of shape (num_points, dim).
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
                result[:, i] = geodesic_equation(y[:, i], lambda x: metric_s, eps=eps)
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

    s = 0.0
    ds = 1.0 / n_steps
    while s < 1.0 - 1e-12:
        s_next = min(s + ds, 1.0)
        sol = solve_bvp(make_fun(s_next, pbar), bc, t_cur, y_cur, tol=tol)

        if sol.success:
            # Accept step: use refined mesh as next warm start
            t_cur = sol.x
            y_cur = sol.y
            s = s_next
        else:
            # Halve step and retry
            ds /= 2
            if ds < 1e-6:
                raise RuntimeError(f"Continuation failed at s={s:.4f}: step size too small")

    if pbar is not None:
        pbar.close()

    t_out = np.linspace(0, 1, num_points)
    return sol.sol(t_out)[:dim].T  # (num_points, dim)


if __name__ == '__main__':
    for dim in [2, 3, 4, 5]:
        def poincare_metric(xx):
            r_squared = np.sum(np.array(xx) ** 2)
            factor = 4.0 / (1 - r_squared)**2
            return factor * np.eye(dim)

        xx = np.random.normal(size=dim)
        xx = xx / np.linalg.norm(xx) * 0.5  # Scale to be within unit disk
        Ric = compute_ricci_curvature(xx, poincare_metric)

        print(f"Dimension: {dim}, Ricci Curvature: {Ric}, Expected: {-dim * (dim - 1)}")

    # Test Ricci scalar integration over mesh in Poincare disk
    print("\n=== Mesh-based Integration Test ===")
    dim = 2

    def poincare_metric_2d(xx):
        r_squared = np.sum(np.array(xx) ** 2)
        if r_squared >= 1.0:
            raise ValueError("Outside Poincare disk")
        factor = 4.0 / (1 - r_squared) ** 2
        return factor * np.eye(dim)

    # Create a simple 2D triangular mesh (unit square divided into 2 triangles)
    mesh_points = np.array([
        [0.0, 0.0],
        [0.3, 0.0],
        [0.0, 0.3],
        [0.3, 0.3]
    ])
    mesh_simplices = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])

    mesh_domain = {'type': 'mesh', 'points': mesh_points, 'simplices': mesh_simplices}
    integral_mesh, error_mesh = integrate_ricci_scalar(poincare_metric_2d, mesh_domain, num_samples=5000, eps=1e-6)

    print(f"Integration over triangular mesh:")
    print(f"  ∫_V R √det(g) d^2x ≈ {integral_mesh:.6f} ± {error_mesh:.6f}")
    print(f"  Mesh volume (Euclidean): {0.3 * 0.3}")
