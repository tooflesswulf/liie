import numpy as np


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
    for k in range(dim):
        for i in range(dim):
            for j in range(i + 1, dim):
                avg = (Gamma[k, i, j] + Gamma[k, j, i]) / 2
                Gamma[k, i, j] = avg
                Gamma[k, j, i] = avg

    return Gamma

def compute_christoffel_symbols_analytic(x, metric, metric_jacobian):
    """
    Compute Christoffel symbols of the second kind using an analytic metric Jacobian.

    Γ^k_{ij} = (1/2) g^{kl} (∂g_{jl}/∂x^i + ∂g_{il}/∂x^j - ∂g_{ij}/∂x^l)

    Parameters
    ----------
    x : array-like, shape (dim,)
        Point at which to evaluate the Christoffel symbols.
    metric : callable
        x -> (dim, dim) positive-definite metric tensor.
    metric_jacobian : callable
        x -> (dim, dim, dim) array where dg[k, i, j] = ∂g_{ij}/∂x^k.

    Returns
    -------
    Gamma : (dim, dim, dim) array
        Gamma[k, i, j] = Γ^k_{ij}, symmetric in the lower two indices.
    """
    g = metric(x)
    g_inv = np.linalg.inv(g)
    dg = metric_jacobian(x)   # dg[k, i, j] = ∂g_{ij}/∂x^k

    # Γ^k_{ij} = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
    #           = (1/2) g^{kl} (dg[i,j,l] + dg[j,i,l] - dg[l,i,j])
    Gamma = 0.5 * (
        np.einsum('kl,ijl->kij', g_inv, dg) +   # ∂_i g_{jl}
        np.einsum('kl,jil->kij', g_inv, dg) -   # ∂_j g_{il}
        np.einsum('kl,lij->kij', g_inv, dg)     # ∂_l g_{ij}
    )

    return Gamma


def geodesic_equation(state, metric, eps=1e-5):
    """
    Geodesic equation in terms of Christoffel symbols:
    d^2 x^k / dt^2 + Γ^k_{ij} (dx^i/dt) (dx^j/dt) = 0

    State vector: [x^1, ..., x^n, dx^1/dt, ..., dx^n/dt]

    Returns the time derivative: [dx^i/dt, d^2x^i/dt^2]
    """
    dim = len(state) // 2
    x = state[:dim]
    xdot = state[dim:]
    Gamma = compute_christoffel_symbols(x, metric, eps=eps)

    xddot = np.zeros(dim)
    for k in range(dim):
        xddot[k] = -np.sum(Gamma[k, :, :] * np.outer(xdot, xdot))

    return np.r_[xdot, xddot]
