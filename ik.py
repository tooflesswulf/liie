# Full-arm IK
import numpy as np
import pinocchio as pin

import build_robot
import robot


# def ik(from_arm: robot.RobotArm, to_arm: robot.RobotArm, from_q0, to_q):
#     to_arm.update(to_q)
#     dt = .1

#     q0 = np.copy(from_q0)
#     for i in range(1000):
#         from_arm.update(q0)

#         errs = []
#         Js = []
#         for i in range(1, 11):
#             iMd = from_arm.data.oMf[i].actInv(to_arm.data.oMf[i])

#             erri = pin.log(iMd).vector
#             erri[3:] = 0
#             # erri[:3] = 0
#             Ji = pin.computeFrameJacobian(from_arm.model, from_arm.data, q0, i, pin.LOCAL)
#             Ji = -np.dot(pin.Jlog6(iMd.inverse()), Ji)
#             errs.append(erri)
#             Js.append(Ji)

#         err = np.hstack(errs)
#         J = np.vstack(Js)
#         v = -J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(len(J)), err)
#         q0 = q0 + v * dt
#     return q0

#     # print(np.hstack(errs).shape)
#     # print(np.vstack(Js).shape)

#         # v = - Ji.T.dot(np.linalg.solve(Ji.dot(Ji.T) + damp * np.eye(6), erri))
#         # print(erri)
#         # print(Ji.shape)
        
def ik(from_arm: robot.RobotArm, to_arm: robot.RobotArm, from_q0, to_q,
       n_iters=1000, dt=0.1, ee_only_iters=400, blend_iters=600):
    """
    Two-phase IK: end-effector first, then blend in intermediate frames.
    
    Phase 1 (0 to ee_only_iters): only match end-effector (frame 10)
    Phase 2 (ee_only_iters to ee_only_iters+blend_iters): linearly ramp up
            intermediate frame weights from 0 to 1
    Phase 3 (remaining): full-link IK with all frames equally weighted
    """
    to_arm.update(to_q)
    q0 = np.copy(from_q0)

    ee_frame = 10  # last frame index

    for it in range(n_iters):
        from_arm.update(q0)

        # Compute per-frame weights based on current phase
        if it < ee_only_iters:
            # Phase 1: end-effector only
            intermediate_weight = 0.0
        elif it < ee_only_iters + blend_iters:
            # Phase 2: linear blend
            intermediate_weight = (it - ee_only_iters) / blend_iters
        else:
            # Phase 3: full weight
            intermediate_weight = 1.0

        errs = []
        Js = []
        for f in range(1, 11):
            iMd = from_arm.data.oMf[f].actInv(to_arm.data.oMf[f])
            erri = pin.log(iMd).vector
            erri[3:] = 0  # position only

            Ji = pin.computeFrameJacobian(
                from_arm.model, from_arm.data, q0, f, pin.LOCAL)
            Ji = -np.dot(pin.Jlog6(iMd.inverse()), Ji)

            # Weight: end-effector always 1.0, intermediates ramp up
            w = 1.0 if f == ee_frame else intermediate_weight

            errs.append(w * erri)
            Js.append(w * Ji)

        err = np.hstack(errs)
        J = np.vstack(Js)

        # Early termination
        if np.linalg.norm(err) < 1e-6:
            break

        v = -J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(len(J)), err)
        q0 = q0 + v * dt

    return q0


if __name__ == '__main__':
    ll1 = [0, 1/3, 1/3, 1/3]
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

    model1, geom_model1 = build_robot.build_simple_arm(ll1, axes1, visualize=True)
    arm1 = robot.RobotArm(model1, geom_model1, Nintermediate=11)

    model2, geom_model2 = build_robot.build_simple_arm(
        ll2, axes2, visualize=True, prefix='arm2_', link_color=(1.0, 0.5, 0.055, 1.0))
    arm2 = robot.RobotArm(model2, geom_model2, Nintermediate=11, prefix='arm2_')

    q1 = np.array([1, .3, 1.3, .4])
    # q2 = np.zeros(model2.nq)

    # arm1.display_state(q1)
    # import time
    # time.sleep(2)
    # exit(0)


    # Joint visualizer
    m, g = build_robot.build_simple_arm(ll1, axes1, visualize=True)
    build_robot.build_simple_arm(ll2, axes2, model=m, geom_model=g, visualize=True, link_color=(1.0, 0.5, 0.055, 1.0))
    a = robot.RobotArm(m, g)

    import time
    while True:
        q2 = np.random.random(model2.nq) - .5
        # q2 = np.array([1.01299624, 1.02142395, 0])
        nq2 = ik(arm2, arm1, q2, q1)
        a.display_state(np.r_[q1, nq2])
        # print(nq2)
        time.sleep(.5)

    # import time
    # time.sleep(1)

    # data = model.createData()
    # robot.add_intermediate_frames(arm, model, geom_model, N=11)
    # viz = build_robot.setup_visualizer(model, geom_model)

    # pin.forwardKinematics(model, arm.data, q)
    # pin.updateFramePlacements(model, arm.data)
    # print(len(arm.data.oMf))
    # print(list(arm.data.oMf))

    # print(pin.computeFrameJacobian(model, data, q, 6))
