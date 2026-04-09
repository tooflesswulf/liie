import npoly2d as n2

import numpy as np
import scipy.special
from scipy.integrate import solve_ivp
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

def integrate_ricci_scalar(metric, domain, num_samples=1000, eps=1e-5):
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
    for sample in samples:
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


def compute_geodesic(x0, v0, metric, length=1.0, num_points=100, eps=1e-5):
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

    # Integrate geodesic equation
    sol = solve_ivp(
        lambda t, y: geodesic_equation(y, metric),
        [0, length],
        y0,
        t_eval=np.linspace(0, length, num_points),
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )

    return sol.y[:dim, :].T  # Return only positions


def dist_geo(x1, x2, metric, eps=1e-5):
    """Compute geodesic distance between two points using numerical integration."""
    # Compute geodesic connecting x1 and x2
    direction = x2 - x1
    length = np.linalg.norm(direction)
    geodesic = compute_geodesic(x1, direction, metric, length=length, num_points=100, eps=eps)

    # # Compute length of geodesic by integrating √(g_{ij} dx^i/dt dx^j/dt)
    # total_length = 0.0
    # for i in range(len(geodesic) - 1):
    #     mid_point = (geodesic[i] + geodesic[i + 1]) / 2
    #     g_mid = metric(mid_point)
    #     dx = geodesic[i + 1] - geodesic[i]
    #     ds = np.sqrt(dx @ g_mid @ dx)
    #     total_length += ds

    # return total_length


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
