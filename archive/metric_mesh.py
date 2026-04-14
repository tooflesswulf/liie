import npoly2d as n2

import numpy as np
from scipy.integrate import solve_ivp
import yaml
import sympy as sym
import matplotlib.pyplot as plt


def compute_christoffel_symbols(x, y, metric, eps=1e-5):
    """
    Compute Christoffel symbols of the second kind via finite differences.
    Γ^k_{ij} = (1/2) g^{kl} (∂g_{jl}/∂x^i + ∂g_{il}/∂x^j - ∂g_{ij}/∂x^l)
    """
    g = metric(x, y)
    g_inv = np.linalg.inv(g)

    # Compute metric derivatives via finite differences
    dg_dx = (metric(x + eps, y) - metric(x - eps, y)) / (2 * eps)
    dg_dy = (metric(x, y + eps) - metric(x, y - eps)) / (2 * eps)

    Gamma = np.zeros((2, 2, 2))  # Gamma[k, i, j]

    for k in range(2):
        for i in range(2):
            for j in range(2):
                dg_derivatives = [dg_dx, dg_dy]
                christoffel_sum = 0.0
                for l in range(2):
                    christoffel_sum += g_inv[k, l] * (
                        dg_derivatives[i][j, l] +
                        dg_derivatives[j][i, l] -
                        dg_derivatives[l][i, j]
                    )
                Gamma[k, i, j] = 0.5 * christoffel_sum

    # Ensure symmetry in lower indices: Γ^k_{ij} = Γ^k_{ji}
    for k in range(2):
        for i in range(2):
            for j in range(i + 1, 2):
                avg = (Gamma[k, i, j] + Gamma[k, j, i]) / 2
                Gamma[k, i, j] = avg
                Gamma[k, j, i] = avg

    return Gamma


def compute_gaussian_curvature(x, y, metric, eps=1e-5):
    """
    Compute the Gaussian curvature at a point using the Riemann curvature tensor.

    For a 2D surface, Gaussian curvature K is given by:
    K = R_1212 / det(g)

    where R_1212 is the only independent component of the Riemann tensor.

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
    g = metric(x, y)
    det_g = np.linalg.det(g)

    # Get Christoffel symbols at the point and neighboring points
    Gamma = compute_christoffel_symbols(x, y, metric, eps)
    Gamma_dx_plus = compute_christoffel_symbols(x + eps, y, metric, eps)
    Gamma_dx_minus = compute_christoffel_symbols(x - eps, y, metric, eps)
    Gamma_dy_plus = compute_christoffel_symbols(x, y + eps, metric, eps)
    Gamma_dy_minus = compute_christoffel_symbols(x, y - eps, metric, eps)

    # Compute derivatives of Christoffel symbols
    dGamma_dx = (Gamma_dx_plus - Gamma_dx_minus) / (2 * eps)
    dGamma_dy = (Gamma_dy_plus - Gamma_dy_minus) / (2 * eps)

    # For 2D surfaces, use the explicit formula for Gaussian curvature
    # The Riemann curvature tensor in 2D has only one independent component
    # K = (1/det(g)) * [∂_y Γ^x_{xy} - ∂_x Γ^y_{xy} + Γ^x_{xy}Γ^y_{xx} - Γ^x_{xx}Γ^y_{xy}
    #                   + Γ^x_{yy}Γ^y_{xy} - Γ^x_{xy}Γ^y_{yy}]
    #
    # Or equivalently using R^x_{yxy}:
    # R^x_{yxy} = ∂_x Γ^x_{yy} - ∂_y Γ^x_{xy} + Γ^x_{mx}Γ^m_{yy} - Γ^x_{my}Γ^m_{xy}

    # Method 1: Compute R^0_{101} (R^x_{yxy} in index notation)
    # R^0_{101} = ∂_0 Γ^0_{11} - ∂_1 Γ^0_{01} + Γ^0_{m0}Γ^m_{11} - Γ^0_{m1}Γ^m_{01}
    R_0_101 = dGamma_dx[0, 1, 1] - dGamma_dy[0, 0, 1]
    for m in range(2):
        R_0_101 += Gamma[0, m, 0] * Gamma[m, 1, 1] - Gamma[0, m, 1] * Gamma[m, 0, 1]

    # R^1_{101} = ∂_0 Γ^1_{11} - ∂_1 Γ^1_{01} + Γ^1_{m0}Γ^m_{11} - Γ^1_{m1}Γ^m_{01}
    R_1_101 = dGamma_dx[1, 1, 1] - dGamma_dy[1, 0, 1]
    for m in range(2):
        R_1_101 += Gamma[1, m, 0] * Gamma[m, 1, 1] - Gamma[1, m, 1] * Gamma[m, 0, 1]

    # Contract to get scalar curvature: R_0101 = g_{0i} R^i_{101}
    R_0101 = g[0, 0] * R_0_101 + g[0, 1] * R_1_101

    # DEBUG - uncomment to see intermediate values
    if False:  # Set to True to enable debug
        print(f"    Gamma[0,1,1]={Gamma[0, 1, 1]:.6f}, Gamma[0,0,1]={Gamma[0, 0, 1]:.6f}")
        print(f"    dGamma_dx[0,1,1]={dGamma_dx[0, 1, 1]:.6f}, dGamma_dy[0,0,1]={dGamma_dy[0, 0, 1]:.6f}")
        print(f"    R_0_101={R_0_101:.6f}, R_1_101={R_1_101:.6f}, R_0101={R_0101:.6f}")
        print(f"    det_g={det_g:.6f}")

    # Gaussian curvature
    K = R_0101 / det_g if det_g != 0 else 0.0
    return K


def geodesic_equation(t, y, metric):
    """
    Geodesic equation: d²x^k/dt² + Γ^k_{ij} (dx^i/dt)(dx^j/dt) = 0
    State vector: y = [c1, c2, dc1/dt, dc2/dt]
    """
    c1, c2, v1, v2 = y

    Gamma = compute_christoffel_symbols(c1, c2, metric)

    # Acceleration terms
    a1 = -Gamma[0, 0, 0] * v1 * v1 - 2 * Gamma[0, 0, 1] * v1 * v2 - Gamma[0, 1, 1] * v2 * v2
    a2 = -Gamma[1, 0, 0] * v1 * v1 - 2 * Gamma[1, 0, 1] * v1 * v2 - Gamma[1, 1, 1] * v2 * v2

    return [v1, v2, a1, a2]


def compute_geodesic(start, direction, metric, length=1.0, num_points=50):
    """
    Compute a geodesic starting from 'start' in direction 'direction'.
    """
    # Normalize direction according to metric at start point
    g_start = metric(start[0], start[1])
    direction = np.array(direction)
    norm = np.sqrt(direction @ g_start @ direction)
    if norm > 0:
        direction = direction / norm

    # Initial state: position + velocity
    y0 = [start[0], start[1], direction[0], direction[1]]

    # Integrate geodesic equation
    sol = solve_ivp(
        lambda t, y: geodesic_equation(t, y, metric),
        [0, length],
        y0,
        t_eval=np.linspace(0, length, num_points),
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )

    return sol.y[0:2, :].T  # Return [c1, c2] positions


if __name__ == '__main__':
    # Standard Poincare disk metric with K = -1
    def poincare_metric(x, y):
        r_squared = x**2 + y**2
        factor = 4.0 / (1 - r_squared)**2
        return factor * np.eye(2)

    def poincare_christoffel_analytical(x, y):
        """Analytical Christoffel symbols for Poincare disk with factor 4"""
        r_squared = x**2 + y**2
        denom = 1.0 - r_squared

        Gamma = np.zeros((2, 2, 2))
        # For metric g = (4/(1-r²)²) δ_ij
        # Γ^x_{xx} = 2x/(1-r²), Γ^x_{xy} = 2y/(1-r²), Γ^x_{yy} = -2x/(1-r²)
        # Γ^y_{xx} = -2y/(1-r²), Γ^y_{xy} = 2x/(1-r²), Γ^y_{yy} = 2y/(1-r²)
        Gamma[0, 0, 0] = 2 * x / denom
        Gamma[0, 0, 1] = Gamma[0, 1, 0] = 2 * y / denom
        Gamma[0, 1, 1] = -2 * x / denom
        Gamma[1, 0, 0] = -2 * y / denom
        Gamma[1, 0, 1] = Gamma[1, 1, 0] = 2 * x / denom
        Gamma[1, 1, 1] = 2 * y / denom
        return Gamma

    print("Testing Poincare disk curvature (should be K = -1):")
    test_points = [(0.0, 0.0), (0.2, 0.0), (0.0, 0.2), (0.3, 0.3), (0.5, 0.0)]

    for x, y in test_points:
        # Verify Christoffel symbols match
        Gamma_numerical = compute_christoffel_symbols(x, y, poincare_metric, eps=1e-6)
        Gamma_analytical = poincare_christoffel_analytical(x, y)
        max_diff = np.abs(Gamma_numerical - Gamma_analytical).max()

        K = compute_gaussian_curvature(x, y, poincare_metric, eps=1e-6)
        print(f"  Point ({x:.1f}, {y:.1f}): K = {K:.6f}, Γ error = {max_diff:.2e}")
