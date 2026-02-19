import numpy as np
import sympy as sym
# import scipy.spatial.transform.rotation
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import yaml

import pinocchio as pin
import pinocchio.visualize as pv

import build_robot


def Adg(t):
    # Ad_gamma(t) when gamma = [0, 0, t, 0, 0, 0]
    return sym.Matrix([
        [1, 0, 0, 0, -t, 0],
        [0, 1, 0, t, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])


def inner_product(gab, xi1, xi2):
    t, s = sym.symbols('t s')
    gd1 = Adg(s).inv() @ sym.integrate(Adg(t) @ xi1, (t, 0, s))
    gd2 = Adg(s).inv() @ sym.integrate(Adg(t) @ xi2, (t, 0, s))

    return float(sym.integrate((gd1.T @ gab @ gd2)[0, 0], (s, 0, 1)))


def load_coeffs(file):
    with open(file, 'r') as f:
        coeffs_string = yaml.safe_load(f)
    coeffs = [sym.sympify(cs) for cs in coeffs_string]

    def eval_coeffs(x):
        return np.array([ci.subs({sym.symbols('x'): x}) for ci in coeffs]).astype(float)
    return eval_coeffs


if __name__ == '__main__':
    sx = sym.symbols('x')

    # [1,1,0] -> '', [1,1,1] -> '1', [0,0,1] -> '2'
    with open(f'_orthopoly.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp0 = [sym.sympify(ps) for ps in polys_string]
    coeff0 = load_coeffs('_polycoeffs.yaml')

    with open(f'_orthopoly1.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp1 = [sym.sympify(ps) for ps in polys_string]
    coeff1 = load_coeffs('_polycoeffs1.yaml')

    with open(f'_orthopoly2.yaml', 'r') as f:
        polys_string = yaml.safe_load(f)
    pp2 = [sym.sympify(ps) for ps in polys_string]
    coeff2 = load_coeffs('_polycoeffs2.yaml')

    # print(pp0[4])
    gab = np.diag([1, 1, 1, 0, 0, 1])
    xi1 = sym.Matrix([0, 0, 0, pp0[0], 0, pp2[3]])
    xi2 = sym.Matrix([0, 0, 0, pp0[1], 0, pp2[0]])

    print('Inner product:', inner_product(gab, xi1, xi2))

    z = np.zeros(10)

    def axes2coeffs(ll, axes):
        lens = np.cumsum(ll)
        for li, axi in zip(lens, axes):
            if axi == (1, 0, 0):
                yield np.r_[coeff0(li), z, z]
            elif axi == (0, 1, 0):
                yield np.r_[z, coeff0(li), z]
            elif axi == (0, 0, 1):
                yield np.r_[z, z, coeff2(li)]
            else:
                raise ValueError(f"Unsupported axis {axi}")

    ll1 = [0, .33, .33, .33]
    axes1 = [
        (0, 0, 1),  # Joint 1: Revolute around Z-axis
        (0, 1, 0),  # Joint 2: Revolute around Y-axis
        (1, 0, 0),  # Joint 3: Revolute around X-axis
        (0, 0, 1),  # Joint 4: Revolute around Z-axis
    ]

    ll2 = [0, .5, .5]
    axes2 = [
        (0, 0, 1),
        (1, 0, 0),
        (0, 0, 1),
    ]

    mat1 = np.array(list(axes2coeffs(ll1, axes1))).T
    mat2 = np.array(list(axes2coeffs(ll2, axes2))).T

    model, geom_model = build_robot.build_simple_arm(ll1, axes1, visualize=True)
    build_robot.build_simple_arm(ll2, axes2, model=model,
                                 geom_model=geom_model, visualize=True,
                                 link_color=(1.0, 0.5, 0.055, 1.0))
    data = model.createData()
    viz = build_robot.setup_visualizer(model, geom_model)

    q1 = np.array([.5, .6, .7])
    q0 = np.linalg.pinv(mat1) @ mat2 @ q1
    q = np.r_[q0, q1]

    # AI sliders
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.25, bottom=0.35)
    # Hide the main axes
    ax.axis('off')
    # Create sliders
    sliders = []
    slider_axes = []

    for i in range(4):
        # Create axes for slider
        ax_slider = plt.axes([0.25, .8 - i * 0.08, 0.5, 0.03])
        slider = Slider(
            ax_slider,
            f'Joint {i + 1}',
            -np.pi,
            np.pi,
            valinit=0,
            valstep=0.01
        )
        sliders.append(slider)
        slider_axes.append(ax_slider)

    # Update function
    def update(val):
        for i, slider in enumerate(sliders):
            q0[i] = slider.val
            # q1[i] = slider.val

        # q0 = np.linalg.pinv(mat1) @ mat2 @ q1
        q1 = np.linalg.pinv(mat2) @ mat1 @ q0
        q = np.r_[q0, q1]
        build_robot.display_robot_state(model, data, viz, q, show_frames=True)
    for slider in sliders:
        slider.on_changed(update)

    build_robot.display_robot_state(model, data, viz, q, show_frames=True)
    plt.show()

    # import time
    # q1 = np.array([0, 0, 0, 0, .5, .6, .7])
    # t0 = time.time()
    # T = 5.0
    # while True:
    #     # q0 = np.linalg.pinv(mat1) @ mat2 @ q1
    #     # q = np.r_[q0, q1]
    #     build_robot.display_robot_state(model, data, viz, q, show_frames=True)
    #     time.sleep(1 / 20)
