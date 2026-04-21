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


if __name__ == '__main__':
    sx = sym.symbols('x')
    tt = np.linspace(0, 1, 100)
    bases = u3d.Bases3D(gab_diag=[1, 1, 1, 0, 0, 1], emb_size=4)
    bases.set_spacing(tt)

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

    def metric(xx): return bases.metric(xx)

    # Shoot a geodesic from emb1 in the direction of emb2, using the metric
    # norm of (emb2 - emb1) as the arc length so the geodesic travels a
    # comparable distance. This does not enforce arrival at emb2.
    v0 = emb2 - emb1
    g0 = metric(emb1)
    length = float(np.sqrt(v0 @ g0 @ v0))

    geodesic_embs = dn.compute_geodesic(emb1, v0, metric, length=length,
                                        num_points=20, progress=True)

    # Plot arm shapes along the geodesic
    tt = np.linspace(0, 1, 100)
    ax = plt.figure().add_subplot(projection='3d')
    for emb_i in geodesic_embs:
        xi_i = bases.emb2xi(emb_i, tt)
        gamma_i = npoly3d.shape_exp(xi_i)
        ax.plot(gamma_i[:, 0, 3], gamma_i[:, 1, 3], gamma_i[:, 2, 3], alpha=0.4)
    ax.set_box_aspect([1, 1, 1])
    plt.show()
