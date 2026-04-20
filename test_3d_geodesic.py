import numpy as np
import sympy as sym
import yaml
from scipy.optimize import root
from scipy.integrate import solve_ivp

import bases_fns.util3d as u3d
import bases_fns.npoly3d as npoly3d
import robot_pino.build_robot as build_robot
import robot_pino.robot as robot
import diffgeo_nd as dn


def load_coeffs(file):
    with open(file, 'r') as f:
        coeffs_string = yaml.safe_load(f)
    coeffs = [sym.sympify(cs) for cs in coeffs_string]

    def eval_coeffs(x):
        return np.array([ci.subs({sym.symbols('x'): x}) for ci in coeffs]).astype(float)
    return eval_coeffs


if __name__ == '__main__':
    sx = sym.symbols('x')
    bases = u3d.Bases3D(gab_diag=[1, 1, 1, 0, 0, 1], emb_size=4)

    ll1 = [0, .33, .33, .33]
    axes1 = [
        (0, 0, 1),  # Joint 1: Revolute around Z-axis
        (0, 1, 0),  # Joint 2: Revolute around Y-axis
        (1, 0, 0),  # Joint 3: Revolute around X-axis
        (0, 0, 1),  # Joint 4: Revolute around Z-axis
    ]

    model1, geom_model1 = build_robot.build_simple_arm(ll1, axes1, visualize=True)
    arm1 = robot.RobotArm(model1, geom_model1, Nintermediate=11)

    coeffs1 = bases.embed(ll1, axes1)
    joints1 = np.array([.2, -.5, .5, 0])

    ########################################
    # Stage 1: Visualize exact joint space representation vs approx. shape repr

    # Exact representation
    arm1.update(joints1)
    joint_frames = arm1.data.oMi
    joint_locs = np.array([jf.translation for jf in joint_frames])

    # Approximate representation (first 3 coeffs of each axis)
    emb = joints1 @ coeffs1
    tt = np.linspace(0, 1, 100)
    xi = bases.emb2xi(emb, tt)
    gamma = npoly3d.shape_exp(xi)

    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(gamma[:, 0, 3], gamma[:, 1, 3], gamma[:, 2, 3])
    ax.plot(joint_locs[:, 0], joint_locs[:, 1], joint_locs[:, 2])
    ax.set_box_aspect([1, 1, 1])
    plt.show()


    ########################################
    # Stage 2: Visualize geodesic interpolation between two joint configurations

    ll2 = [0, .5, .5]
    axes2 = [
        (0, 0, 1),
        (1, 0, 0),
        (0, 0, 1),
    ]

    coeffs2 = bases.embed(ll2, axes2)
    model2, geom_model2 = build_robot.build_simple_arm(ll2, axes2, visualize=True)
    arm2 = robot.RobotArm(model2, geom_model2, Nintermediate=11)

    joints2 = np.array([-.5, .5, 0])
    emb1 = joints1 @ coeffs1
    emb2 = joints2 @ coeffs2

    from tqdm.auto import tqdm

    metric = lambda xx: bases.metric(xx, N=20)

    # Shooting: find initial velocity v0 at emb1 s.t. geodesic reaches emb2 at t=1
    n_emb = len(emb1)
    def make_tracked_eq(pbar):
        last_t = [0.0]
        def eq(t, y):
            inc = int((t - last_t[0]) * 100)
            if inc > 0:
                pbar.update(inc)
                last_t[0] = t
            return dn.geodesic_equation(y, metric)
        return eq

    def shoot(v0, pbar):
        pbar.reset()
        y0 = np.r_[emb1, v0]
        sol = solve_ivp(
            make_tracked_eq(pbar),
            [0.0, 1.0], y0, method='RK45', rtol=1e-6, atol=1e-8,
        )
        return sol.y[:n_emb, -1] - emb2

    v0_guess = emb2 - emb1
    with tqdm(total=100, desc='Shooting') as pbar:
        shoot_result = root(lambda v0: shoot(v0, pbar), v0_guess)
    v0 = shoot_result.x

    # Integrate geodesic with enough points for visualization
    y0 = np.r_[emb1, v0]
    with tqdm(total=100, desc='Integrating geodesic') as pbar:
        geo_sol = solve_ivp(
            make_tracked_eq(pbar),
            [0.0, 1.0], y0,
            t_eval=np.linspace(0, 1, 20),
            method='RK45', rtol=1e-6, atol=1e-8,
        )
    geodesic_embs = geo_sol.y[:n_emb, :].T  # (20, n_emb)

    # Plot arm shapes along the geodesic
    tt = np.linspace(0, 1, 100)
    ax = plt.figure().add_subplot(projection='3d')
    for emb_i in geodesic_embs:
        xi_i = bases.emb2xi(emb_i, tt)
        gamma_i = npoly3d.shape_exp(xi_i)
        ax.plot(gamma_i[:, 0, 3], gamma_i[:, 1, 3], gamma_i[:, 2, 3], alpha=0.4)
    ax.set_box_aspect([1, 1, 1])
    plt.show()
