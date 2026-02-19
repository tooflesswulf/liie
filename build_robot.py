import numpy as np
import pinocchio as pin
import pinocchio.visualize
import time


def make_cylinder(name, jid, radius, length, color=(0.2, 0.6, 1.0, 1.0)):
    # Place cylinder at midpoint of link
    location = pin.SE3(np.eye(3), np.array([0, 0, length / 2]))

    # Create a cylinder along the Z-axis
    cylinder = pin.GeometryObject(
        name, jid,
        pin.hppfcl.Cylinder(radius, length),
        location
    )
    cylinder.meshColor = np.array(color)
    return cylinder


def build_simple_arm(lengths, joint_axes=None, visualize=False, model=None, geom_model=None,
                     link_color=(0.2, 0.6, 1.0, 1.0)):
    """
    Build a simple robot arm with joints placed arbitrarily.

    Parameters:
    -----------
    lengths: list of floats
        List of link lengths. The number of joints will be len(lengths).
        Example: [0.5, 0.3] for a 2-link arm with lengths 0.5 and 0.3 units.

    joint_axes : list of tuples/arrays, optional
        List of rotation axes for each joint. Each axis is a (x, y, z) unit vector.
        Default is Z-axis [0, 0, 1] for all joints (revolute joints around Z).
        Example: [(0, 0, 1), (0, 1, 0), (0, 0, 1)]  # Z, Y, Z axes

    visualize : bool, optional
        If True, initialize a MeshCat visualizer and load the model.

    Returns:
    --------
    model : pin.Model
        The Pinocchio model
    data : pin.Data
        The Pinocchio data
    geom_model : pin.GeometryModel
        The geometry model for visualization
    visualizer : pin.visualize.MeshcatVisualizer or None
        The visualizer if visualize=True, else None
    """

    # Initialize model and geometry model
    if model is None:
        model = pin.Model()
        geom_model = pin.GeometryModel()
    if geom_model is None:
        raise ValueError("geom_model must be provided if model is provided")

    # Default joint axes to Z-axis if not provided
    if joint_axes is None:
        joint_axes = [(0, 0, 1)] * len(lengths)

    assert len(joint_axes) == len(lengths), \
        "Number of joint axes must match number of joint positions"

    # Add a base/world frame
    # The first joint is placed relative to this frame
    parent_id = 0  # universe frame

    for i, (ll, axis) in enumerate(zip(lengths, joint_axes)):
        ix = model.nq + 1
        joint_name = f"joint_{ix}"

        # Convert to numpy arrays
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # normalize

        # Joint placement relative to parent frame
        joint_placement = pin.SE3(np.eye(3), np.array([0, 0, ll]))

        # Create inertia (simple uniform density cylinder/box for visualization)
        mass = 1.0
        lever = 0.1  # radius of gyration
        inertia = pin.Inertia(
            mass,
            np.array([0.0, 0.0, 0.0]),  # center of mass in local frame
            mass * lever * np.eye(3)
        )

        # Add revolute joint
        joint_id = model.addJoint(
            parent_id,
            pin.JointModelRY() if np.allclose(axis, [0, 1, 0]) else
            pin.JointModelRX() if np.allclose(axis, [1, 0, 0]) else
            pin.JointModelRZ(),  # default to Z-axis
            joint_placement,
            joint_name
        )

        # Add body/link
        body_placement = pin.SE3.Identity()
        model.appendBodyToJoint(joint_id, inertia, body_placement)

        # Add visual geometry for this link
        if ll > 0:
            cyl = make_cylinder(f"link_{ix}_visual", parent_id, 0.02, ll, color=link_color)
            geom_model.addGeometryObject(cyl)

        # Add a small sphere at the joint location for visualization
        sphere_radius = 0.03
        sphere = pin.GeometryObject(
            f"joint_{ix}_sphere",
            joint_id,
            pin.hppfcl.Sphere(sphere_radius),
            pin.SE3.Identity()
        )
        sphere.meshColor = np.array([1.0, 0.3, 0.3, 1.0])
        geom_model.addGeometryObject(sphere)

        # Update parent for next iteration
        parent_id = joint_id

    return model, geom_model


def setup_visualizer(model, geom_model):
    try:
        visualizer = pin.visualize.MeshcatVisualizer(
            model,
            geom_model,
            geom_model
        )
        visualizer.initViewer(open=True)
        visualizer.loadViewerModel()
        return visualizer
    except Exception as e:
        print(f"Could not initialize visualizer: {e}")
        return None


def display_robot_state(model, data, visualizer, q, show_frames=True):
    """
    Display robot at a given configuration.

    Parameters:
    -----------
    model : pin.Model
    data : pin.Data
    visualizer : pin.visualize.MeshcatVisualizer
    q : array-like
        Joint configuration (angles in radians)
    show_frames : bool, optional
        Whether to display coordinate frames at each joint
    """
    q = np.array(q, dtype=float)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    if visualizer is not None:
        visualizer.display(q)

        # Display frames at each joint
        if show_frames:
            import meshcat.geometry as g
            for i in range(1, model.njoints):
                frame_name = f"joint_{i}_frame"
                try:
                    # Create frame if it doesn't exist
                    visualizer.viewer[frame_name].set_object(
                        g.triad(scale=0.1)
                    )
                    visualizer.viewer[frame_name].set_transform(
                        data.oMi[i].homogeneous
                    )
                except Exception as e:
                    print(f"Could not create frame {frame_name}: {e}")
    return data


# Example usage
if __name__ == '__main__':
    # Example 1: Simple 3-DOF arm with joints on Z-axis
    print("Building 3-DOF arm...")
    lengths = [0, 0.5, 0.4, 0.3]
    axes = [
        (0, 0, 1),  # Joint 1: Revolute around Z-axis
        (0, 1, 0),  # Joint 2: Revolute around Y-axis
        (1, 0, 0),  # Joint 3: Revolute around X-axis
        (0, 0, 1),  # Joint 4: Revolute around Z-axis
    ]

    model, geom_model, viz = build_simple_arm(lengths, axes, visualize=True)
    data = model.createData()

    q0 = 0 * np.array([1, 1, 0, 0])
    qd = .1 * np.array([1, 1, 0., 0])
    t0 = time.time()
    T = 5.0
    while True:
        q = q0 + qd * np.sin(2 * np.pi * (time.time() - t0) / T)
        display_robot_state(model, data, viz, q, show_frames=True)
        time.sleep(1 / 20)

    display_robot_state(model, data, viz, [0.5, -0.5, 0.3], show_frames=True)

    print(f"Model has {model.nq} DOFs")
    print(f"Joint names: {[model.names[i] for i in range(1, model.njoints)]}")

    time.sleep(5)

    display_robot_state(model, data, viz, [0, 0, 0], show_frames=True)
