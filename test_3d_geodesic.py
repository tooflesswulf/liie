import numpy as np
import sympy as sym
import yaml
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pickle

import bases_fns.util3d as u3d
import bases_fns.npoly3d as npoly3d
import robot_pino.build_robot as build_robot
import robot_pino.robot as robot
import diffgeo_nd as dn


def equalize_3d_axes(ax):
    """
    Make a 3D axes display equal-length vectors as equal length visually.

    Sets the physical box proportions to match the data ranges so that one
    data unit looks the same size along every axis.  Unlike expanding all
    limits to the same range, this leaves the data view unchanged.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ranges = limits[:, 1] - limits[:, 0]
    ax.set_box_aspect(ranges)


def plot_gamma_frames(ax, gamma, step=10, triads=True, scale=None, alpha=1.0,
                      arm_color='k', lw=1.5):
    """
    Plot an SE(3) path with RGB orientation frames at intermediate points.

    The path is drawn as a line through the translation components of gamma.
    At every ``step``-th point a small triad of arrows is drawn:
        red   → local X axis  (column 0 of R)
        green → local Y axis  (column 1 of R)
        blue  → local Z axis  (column 2 of R)

    ``equalize_3d_axes(ax)`` is called automatically so that orthogonal
    frame axes appear visually perpendicular regardless of the data range.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
    gamma : (N, 4, 4) array
        SE(3) path, e.g. from ``npoly3d.shape_exp``.
    step : int
        Stride used to subsample frames along the path.
    scale : float or None
        Arrow length in data units.  If None, defaults to 5 % of the total
        arc length of the path.
    alpha : float
        Opacity applied to both the path line and the frame arrows.
    arm_color : color spec
        Colour of the arm curve.
    lw : float
        Line width of the arm curve.

    Returns
    -------
    arc_len : float
        Total arc length of the path (useful for downstream scaling).
    """
    pos     = gamma[:, :3, 3]                              # (N, 3)
    arc_len = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()

    if scale is None:
        scale = 0.05 * arc_len

    # ── Arm curve ─────────────────────────────────────────────────────────────
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            color=arm_color, lw=lw, alpha=alpha)

    if triads:
        # ── RGB frame triads ──────────────────────────────────────────────────────
        idx     = np.arange(0, len(gamma), step)
        origins = gamma[idx, :3, 3]      # (M, 3)  frame origins
        R_mats  = gamma[idx, :3, :3]     # (M, 3, 3)  rotation matrices

        for col, color in enumerate(['r', 'g', 'b']):
            dirs = R_mats[:, :, col]     # (M, 3)  axis direction in world frame
            ax.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
                    dirs[:, 0] * scale, dirs[:, 1] * scale, dirs[:, 2] * scale,
                    color=color, arrow_length_ratio=0.2, alpha=alpha)

    equalize_3d_axes(ax)
    return arc_len


def test1_approx_shape_repr():
    # Test 1: Visualize exact joint space representation vs approx. shape repr

    # Exact representation
    arm1.update(joints1)
    joint_frames = arm1.data.oMi
    joint_locs = np.array([jf.translation for jf in joint_frames])

    # Approximate representation (first 3 coeffs of each axis)
    emb = joints1 @ coeffs1
    tt = np.linspace(0, 1, 100)
    xi = bases.emb2xi(emb, tt)
    gamma = npoly3d.shape_exp(xi)

    ax = plt.figure().add_subplot(projection='3d')
    plot_gamma_frames(ax, gamma, step=10)
    # ax.plot(gamma[:, 0, 3], gamma[:, 1, 3], gamma[:, 2, 3])
    ax.plot(joint_locs[:, 0], joint_locs[:, 1], joint_locs[:, 2])
    equalize_3d_axes(ax)
    plt.show()

def test2_geodesic_ode():
    # Test 2: Shoot geodesic out from emb1 and see where it lands
    emb1 = joints1 @ coeffs1
    emb2 = joints2 @ coeffs2

    v0 = emb2 - emb1
    g0 = bases.metric(emb1)
    length = float(np.sqrt(v0 @ g0 @ v0))

    geodesic_embs = pickle.load(open('geodesic_ode.pkl', 'rb'))
    # geodesic_embs = dn.compute_geodesic(emb1, v0, bases.metric, length=length,
    #                                     num_points=10, progress=True)
    # pickle.dump(geodesic_embs, open('geodesic_ode.pkl', 'wb'))

    # Plot arm shapes along the geodesic
    tt = np.linspace(0, 1, 100)
    ax = plt.figure().add_subplot(projection='3d')
    for emb_i in geodesic_embs:
        xi_i = bases.emb2xi(emb_i, tt)
        gamma_i = npoly3d.shape_exp(xi_i)
        ax.plot(gamma_i[:, 0, 3], gamma_i[:, 1, 3], gamma_i[:, 2, 3], alpha=0.4)

    # Place initial & target exact shapes for reference
    arm1.update(joints1)
    joint_locs1 = np.array([jf.translation for jf in arm1.data.oMi])
    arm2.update(joints2)
    joint_locs2 = np.array([jf.translation for jf in arm2.data.oMi])
    ax.plot(joint_locs1[:, 0], joint_locs1[:, 1], joint_locs1[:, 2], 'k.-')
    ax.plot(joint_locs2[:, 0], joint_locs2[:, 1], joint_locs2[:, 2], 'r.-')
    equalize_3d_axes(ax)
    plt.show()

def test3_geodesic_bvp():
    # Test 3: Solve BVP for geodesic between emb1 and emb2, compare to ODE solution
    emb1 = joints1 @ coeffs1
    emb2 = joints2 @ coeffs2

    # geodesic_embs = dn.geodesic_bvp_continuation(emb1, emb2, bases.metric, num_points=20,
    #                                              n_steps=50, progress=True, max_points=50)
    # pickle.dump(geodesic_embs, open('geodesic_bvp.pkl', 'wb'))
    geodesic_embs = pickle.load(open('geodesic_bvp.pkl', 'rb'))
    straight_embs = emb1[None,:] + np.linspace(0, 1, 20)[:,None]*(emb2-emb1)

    fig = plt.figure()

    # Plot arm shapes along the geodesic
    tt = np.linspace(0, 1, 100)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for emb_i in geodesic_embs:
        xi_i = bases.emb2xi(emb_i, tt)
        gamma_i = npoly3d.shape_exp(xi_i)
        ax.plot(gamma_i[:, 0, 3], gamma_i[:, 1, 3], gamma_i[:, 2, 3], alpha=0.4)

    # Place initial & target exact shapes for reference
    arm1.update(joints1)
    joint_locs1 = np.array([jf.translation for jf in arm1.data.oMi])
    arm2.update(joints2)
    joint_locs2 = np.array([jf.translation for jf in arm2.data.oMi])
    ax.plot(joint_locs1[:, 0], joint_locs1[:, 1], joint_locs1[:, 2], 'k.-')
    ax.plot(joint_locs2[:, 0], joint_locs2[:, 1], joint_locs2[:, 2], 'r.-')
    equalize_3d_axes(ax)

    # Plot arm shapes along a straight line
    tt = np.linspace(0, 1, 100)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for emb_i in straight_embs:
        xi_i = bases.emb2xi(emb_i, tt)
        gamma_i = npoly3d.shape_exp(xi_i)
        ax.plot(gamma_i[:, 0, 3], gamma_i[:, 1, 3], gamma_i[:, 2, 3], alpha=0.4)

    # Place initial & target exact shapes for reference
    arm1.update(joints1)
    joint_locs1 = np.array([jf.translation for jf in arm1.data.oMi])
    arm2.update(joints2)
    joint_locs2 = np.array([jf.translation for jf in arm2.data.oMi])
    ax.plot(joint_locs1[:, 0], joint_locs1[:, 1], joint_locs1[:, 2], 'k.-')
    ax.plot(joint_locs2[:, 0], joint_locs2[:, 1], joint_locs2[:, 2], 'r.-')
    equalize_3d_axes(ax)
    plt.show()


def _tangent_to_arrows(emb, v, bases, tt):
    """
    Lift a tangent vector v at embedding point emb to a 3D deformation
    field along the arm.

    The body-frame variation η_v = Ad(γ)^{-1} ∫ Ad(γ) ξ_v dt splits into
    a translational part (η_v[:3]) and a rotational part (η_v[3:]).
    Both are rotated to the world frame via R(s).

    Returns
    -------
    pos       : (N, 3)  world positions along the arm
    delta_pos : (N, 3)  world-frame translational deformation  R @ η_v[:3]
    delta_rot : (N, 3)  world-frame rotation-axis field        R @ η_v[3:]
    """
    xi0   = bases.emb2xi(emb, tt)
    gamma = npoly3d.shape_exp(xi0)
    pos   = gamma[:, :3, 3]
    R     = gamma[:, :3, :3]

    xi_v      = bases.emb2xi(v, tt)
    eta_v     = npoly3d.gammadot(gamma, xi_v)          # (N, 6) body-frame variation
    delta_pos = np.einsum('nij,nj->ni', R, eta_v[:, :3])  # translational component
    delta_rot = np.einsum('nij,nj->ni', R, eta_v[:, 3:])  # rotational component
    return pos, delta_pos, delta_rot


def _quiver_tangent(ax, pos, delta_pos, delta_rot, arm_len, idx,
                    color, show_rotation, label=None):
    """
    Draw translational and (optionally) rotational arrows for one tangent vector.

    Translational arrows are solid, scaled to 10 % of arm length.
    Rotational arrows are drawn as shorter dashed stubs from the same base
    points, with a square marker at the tip to distinguish them visually.
    They share the same colour but at reduced alpha so they don't overpower
    the position arrows.
    """
    # ── Translational arrows ─────────────────────────────────────────────────
    t_norm = np.linalg.norm(delta_pos, axis=1).mean() + 1e-12
    scale  = 0.10 * arm_len / t_norm
    ax.quiver(pos[idx, 0], pos[idx, 1], pos[idx, 2],
              delta_pos[idx, 0] * scale,
              delta_pos[idx, 1] * scale,
              delta_pos[idx, 2] * scale,
              color=color, arrow_length_ratio=0.3, label=label)

    # ── Rotational arrows (optional) ─────────────────────────────────────────
    if show_rotation:
        r_norm  = np.linalg.norm(delta_rot, axis=1).mean() + 1e-12
        r_scale = 0.07 * arm_len / r_norm   # slightly shorter than position arrows
        ax.quiver(pos[idx, 0], pos[idx, 1], pos[idx, 2],
                  delta_rot[idx, 0] * r_scale,
                  delta_rot[idx, 1] * r_scale,
                  delta_rot[idx, 2] * r_scale,
                  color=color, arrow_length_ratio=0.5, alpha=0.45,
                  linestyle='dashed')


def test4_tangent_visualization(show_rotation=True):
    """
    Test 4: Visualize tangent vectors in the embedding / shape space as
    deformation arrow fields along the 3D arm curve.

    The body-frame variation η_v = Ad(γ)^{-1} ∫ Ad(γ) ξ_v dt carries both
    a translational part (position shift of each cross-section) and a
    rotational part (instantaneous rotation axis of each cross-section):

        δp(s)   = R(s) @ η_v(s)[:3]   — solid arrows
        δω(s)   = R(s) @ η_v(s)[3:]   — dashed arrows (if show_rotation=True)

    Two subplots are shown:
      Left  — the geodesic direction (emb2 − emb1) as a single field
      Right — the first few embedding basis vectors e_k, one colour each

    Parameters
    ----------
    show_rotation : bool
        When True, draw the rotational component as shorter dashed arrows
        alongside the translational (solid) arrows.
    """
    tt    = np.linspace(0, 1, 100)
    emb1  = joints1 @ coeffs1
    emb2  = joints2 @ coeffs2
    step  = 8

    fig = plt.figure(figsize=(14, 6))

    # ── Left: geodesic velocity direction ────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    v_geo                    = emb2 - emb1
    pos, delta_pos, delta_rot = _tangent_to_arrows(emb1, v_geo, bases, tt)
    arm_len = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()
    idx     = np.arange(0, len(tt), step)

    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', lw=2)
    _quiver_tangent(ax1, pos, delta_pos, delta_rot, arm_len, idx,
                    color='r', show_rotation=show_rotation)

    title1 = 'Geodesic direction (emb2 − emb1)'
    if show_rotation:
        title1 += '\nsolid = translation  ╌╌ = rotation axis'
    ax1.set_title(title1, fontsize=9)
    equalize_3d_axes(ax1)

    # ── Right: individual embedding basis vectors e_k ─────────────────────
    ax2    = fig.add_subplot(1, 2, 2, projection='3d')
    n_show = min(6, len(emb1))
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))

    ax2.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'k-', lw=1.5, alpha=0.4)
    for k in range(n_show):
        e_k = np.zeros_like(emb1); e_k[k] = 1.0
        _, dp_k, dr_k = _tangent_to_arrows(emb1, e_k, bases, tt)
        _quiver_tangent(ax2, pos, dp_k, dr_k, arm_len, idx,
                        color=colors[k], show_rotation=show_rotation,
                        label=f'$e_{{{k}}}$')

    title2 = 'Embedding basis vectors $e_k$'
    if show_rotation:
        title2 += '\nsolid = translation  ╌╌ = rotation axis'
    ax2.set_title(title2, fontsize=9)
    equalize_3d_axes(ax2)
    ax2.legend(fontsize=8, loc='upper left')

    plt.suptitle('Tangent vectors in shape/embedding space → 3D arm deformations')
    plt.tight_layout()
    plt.show()


def test5_tangent_transport():
    # Test 5: Parallel transport a tangent vector along the geodesic, see how it changes
    tt    = np.linspace(0, 1, 100)
    emb1  = joints1 @ coeffs1
    emb2  = joints2 @ coeffs2
    step  = 8

    fig = plt.figure(figsize=(14, 6))

    # ── Left: geodesic velocity direction ────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    v_init = [0, 0, 1, 0] @ coeffs1
    pos, delta_pos, delta_rot = _tangent_to_arrows(emb1, v_init, bases, tt)
    arm_len = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()
    idx     = np.arange(0, len(tt), step)

    # Draw start & endpoints
    arm1.update(joints1)
    joint_locs1 = np.array([jf.translation for jf in arm1.data.oMi])
    arm2.update(joints2)
    joint_locs2 = np.array([jf.translation for jf in arm2.data.oMi])
    ax1.plot(joint_locs1[:, 0], joint_locs1[:, 1], joint_locs1[:, 2], 'k.-')
    ax1.plot(joint_locs2[:, 0], joint_locs2[:, 1], joint_locs2[:, 2], 'r.-')

    geodesic_embs = pickle.load(open('geodesic_bvp.pkl', 'rb'))
    for i, emb_i in zip(range(len(geodesic_embs)), geodesic_embs):
        xi_i = bases.emb2xi(emb_i, tt)
        gamma_i = npoly3d.shape_exp(xi_i)
        plot_gamma_frames(ax1, gamma_i, triads=False, alpha=.4, arm_color=f'C{i}')

    # Parallel transport v_geo along the geodesic
    v_pt = dn.parallel_transport(v_init, geodesic_embs, bases.metric)
    pickle.dump(v_pt, open('transported_vector1.pkl', 'wb'))
    # v_pt = pickle.load(open('transported_vector1.pkl', 'rb'))
    pos_pt, dp_pt, dr_pt = _tangent_to_arrows(emb2, v_pt, bases, tt)

    _quiver_tangent(ax1, pos, delta_pos, delta_rot, arm_len/3, idx,
                    color='r', show_rotation=False)
    _quiver_tangent(ax1, pos_pt, dp_pt, dr_pt, arm_len/3, idx,
                    color='g', show_rotation=False)
    equalize_3d_axes(ax1)
    plt.show()


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

    ll2 = [0, .5, .5]
    axes2 = [
        (0, 0, 1),
        (1, 0, 0),
        (0, 0, 1),
    ]
    model2, geom_model2 = build_robot.build_simple_arm(ll2, axes2, visualize=True)
    arm2 = robot.RobotArm(model2, geom_model2, Nintermediate=11)
    coeffs2 = bases.embed(ll2, axes2)
    joints2 = np.array([1, 1.5, 0])

    # test1_approx_shape_repr()
    # test2_geodesic_ode()
    # test3_geodesic_bvp()
    # test4_tangent_visualization(show_rotation=False)
    test5_tangent_transport()
