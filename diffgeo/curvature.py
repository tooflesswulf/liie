import numpy as np
import scipy.special
from tqdm.auto import tqdm

from .symbols import compute_christoffel_symbols


def compute_ricci_curvature(x, metric, eps=1e-5):
    """
    Compute the Ricci scalar at a point via the Riemann curvature tensor.

    R = g^{ij} R_{ij},  where
    R^i_{jkl} = ∂Γ^i_{jl}/∂x^k - ∂Γ^i_{jk}/∂x^l + Γ^m_{jl}Γ^i_{mk} - Γ^m_{jk}Γ^i_{ml}
    """
    dim = len(x)
    g = metric(x)
    ginv = np.linalg.pinv(g)

    Gamma = compute_christoffel_symbols(x, metric, eps)
    dGamma = []
    for i in range(dim):
        dx = np.zeros(dim)
        dx[i] = eps
        dGamma.append(
            (compute_christoffel_symbols(x + dx, metric, eps) -
             compute_christoffel_symbols(x - dx, metric, eps)) / (2 * eps)
        )

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

    return float(np.sum(ginv * R_ij))


# ---------------------------------------------------------------------------
# Monte Carlo integration helpers
# ---------------------------------------------------------------------------

def montecarlo_integration_samples(domain, num_samples=1000):
    """
    Sample points uniformly from a simplex-mesh domain.

    Parameters
    ----------
    domain : dict
        {'type': 'mesh', 'points': (N, dim), 'simplices': (M, dim+1)}
    num_samples : int

    Returns
    -------
    samples : (num_samples, dim)
    total_volume : float
    """
    if domain['type'] != 'mesh':
        raise ValueError(f"Only mesh domains are supported. Got: {domain['type']}")

    points = np.asarray(domain['points'])
    simplices = np.asarray(domain['simplices'])
    dim = points.shape[1]

    simplex_volumes = []
    for simplex in simplices:
        v0 = points[simplex[0]]
        edge_matrix = points[simplex[1:]] - v0
        vol = np.abs(np.linalg.det(edge_matrix)) / scipy.special.factorial(dim)
        simplex_volumes.append(vol)

    simplex_volumes = np.array(simplex_volumes)
    total_volume = np.sum(simplex_volumes)
    simplex_weights = simplex_volumes / total_volume
    sampled_simplices = np.random.choice(len(simplices), size=num_samples, p=simplex_weights)

    samples = np.zeros((num_samples, dim))
    for idx in range(num_samples):
        simplex = simplices[sampled_simplices[idx]]
        vertex_coords = points[simplex]
        bary = np.diff(np.concatenate([[0], np.sort(np.random.uniform(0, 1, dim)), [1]]))
        samples[idx] = np.sum(vertex_coords * bary[:, np.newaxis], axis=0)

    return samples, total_volume


def integrate_ricci_scalar(metric, domain, num_samples=1000, eps=1e-5, progress=False):
    """
    Estimate ∫_V R √det(g) d^n x via Monte Carlo over a simplex-mesh domain.

    Returns
    -------
    integral : float
    error_estimate : float  (1-sigma Monte Carlo standard error)
    """
    samples, volume = montecarlo_integration_samples(domain, num_samples=num_samples)

    integrand_values = []
    iterator = tqdm(samples, desc='Ricci MC', leave=False) if progress else samples
    for sample in iterator:
        try:
            R = compute_ricci_curvature(sample, metric, eps)
            g = metric(sample)
            sqrt_det_g = np.sqrt(np.abs(np.linalg.det(g)))
            integrand_values.append(R * sqrt_det_g)
        except Exception:
            continue

    if not integrand_values:
        raise ValueError("No valid samples in domain.")

    integrand_values = np.array(integrand_values)
    mean = np.mean(integrand_values)
    std = np.std(integrand_values)
    return volume * mean, volume * std / np.sqrt(len(integrand_values))


# ---------------------------------------------------------------------------
# 2D Gauss-Bonnet helpers
# ---------------------------------------------------------------------------

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
    Each cycle is a list of vertex indices in CCW order (interior on the left).
    """
    edge_uses = {}
    for tri in simplices:
        for k in range(3):
            i, j = int(tri[k]), int(tri[(k + 1) % 3])
            edge_uses[(i, j)] = edge_uses.get((i, j), 0) + 1

    next_v = {i: j for (i, j) in edge_uses if (j, i) not in edge_uses}

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
    """+90° rotation of v in the metric g. J(1,0)=(0,1) when g=I."""
    det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
    sd = np.sqrt(det)
    return np.array([
        -(g[0, 1] * v[0] + g[1, 1] * v[1]) / sd,
         (g[0, 0] * v[0] + g[0, 1] * v[1]) / sd,
    ])


def integrate_boundary_term_2d(metric, domain, n_quad=16, eps=1e-5, progress=False):
    """
    Compute the boundary contribution to Gauss-Bonnet on a 2D triangulated domain:

        B(∂M) = ∮_∂M k_g ds  +  Σ_corners θ_i

    Returns
    -------
    kg_integral : float
    corner_sum : float
    """
    points = np.asarray(domain['points'], dtype=float)
    simplices = _orient_ccw_2d(points, domain['simplices'])
    cycles = _boundary_cycles_2d(simplices)

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

        for k in range(N):
            a = points[cycle[k]]
            b = points[cycle[(k + 1) % N]]
            v = b - a
            seg = 0.0
            for t, w in zip(nodes, weights):
                p = a + t * v
                g = metric(p)
                Gamma = compute_christoffel_symbols(p, metric, eps=eps)
                A = np.einsum('kij,i,j->k', Gamma, v, v)
                Jv = _J_rot_2d(g, v)
                s2 = float(v @ g @ v)
                seg += w * float(A @ g @ Jv) / s2
            kg_integral += seg
            if seg_bar is not None:
                seg_bar.update(1)

        for k in range(N):
            p = points[cycle[k]]
            T_in = points[cycle[k]] - points[cycle[(k - 1) % N]]
            T_out = points[cycle[(k + 1) % N]] - points[cycle[k]]
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
    Estimate the Euler characteristic of a 2D Riemannian manifold via Gauss-Bonnet:

        ∫_M K dA + ∮_∂M k_g ds + Σ_corners θ_i = 2π χ(M)

    Returns
    -------
    chi : float
    error : float  (1-sigma Monte Carlo error)
    """
    sample_pt = np.asarray(domain['points'])[0]
    if metric(sample_pt).shape[0] != 2:
        raise NotImplementedError(
            "euler_characteristic is only implemented for 2D manifolds. "
            "For dim > 2 use the Chern-Gauss-Bonnet theorem."
        )

    integral, error = integrate_ricci_scalar(
        metric, domain, num_samples=num_samples, eps=eps, progress=progress
    )
    interior_K = 0.5 * integral  # ∫ K dA = (1/2) ∫ R √g d²x in 2D

    boundary_kg, corner_sum = 0.0, 0.0
    if include_boundary:
        boundary_kg, corner_sum = integrate_boundary_term_2d(
            metric, domain, n_quad=n_quad, eps=eps, progress=progress
        )

    chi = (interior_K + boundary_kg + corner_sum) / (2.0 * np.pi)
    return chi, 0.5 * error / (2.0 * np.pi)


if __name__ == '__main__':
    for dim in [2, 3, 4, 5]:
        def poincare_metric(xx, _dim=dim):
            r_sq = np.sum(np.array(xx) ** 2)
            return 4.0 / (1 - r_sq) ** 2 * np.eye(_dim)

        xx = np.random.normal(size=dim)
        xx = xx / np.linalg.norm(xx) * 0.5
        Ric = compute_ricci_curvature(xx, poincare_metric)
        print(f"dim={dim}: Ricci={Ric:.4f}, expected={-dim * (dim - 1)}")
