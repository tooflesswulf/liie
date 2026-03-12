import pinocchio as pin
import numpy as np
from dataclasses import dataclass

import build_robot


@dataclass
class RobotArm:
    model: pin.Model
    geom_model: pin.GeometryModel
    data: pin.Data
    _viz = None

    link_transforms: list  # List of SE3 transforms for each link
    joint_axes: list       # List of joint axes (e.g., [(0,0,1), (0,1,0), ...])
    prefix: str            # Prefix for joint/frame names

    def __init__(self, model, geom_model, Nintermediate=None, prefix=''):
        self.model = model
        self.geom_model = geom_model

        self.link_transforms = []
        self.joint_axes = []

        # Start from 1 to skip universe joint
        for i in range(1, model.njoints):
            joint = model.joints[i]
            self.link_transforms.append(model.jointPlacements[i])

            if joint.shortname() == 'JointModelRX':
                self.joint_axes.append((1, 0, 0))
            elif joint.shortname() == 'JointModelRY':
                self.joint_axes.append((0, 1, 0))
            elif joint.shortname() == 'JointModelRZ':
                self.joint_axes.append((0, 0, 1))
            else:
                raise ValueError(f"Unsupported joint type: {joint.shortname()}")

        if Nintermediate is not None:
            add_intermediate_frames(self, N=Nintermediate)
        self.data = model.createData()

    @property
    def nq(self):
        return self.model.nq

    def length(self):
        return sum(np.linalg.norm(transform.translation) for transform in self.link_transforms)

    def viz(self):
        if self._viz is None:
            self._viz = build_robot.setup_visualizer(self.model, self.geom_model)
        return self._viz

    def display_state(self, q, show_frames=True):
        build_robot.display_robot_state(self.model, self.data, self.viz(), q, show_frames=show_frames)

    def update(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)


# Util functions
def normalize_arm_length(arm: RobotArm, target_length=1.0):
    total_length = arm.length()
    if total_length == 0:
        return arm  # Avoid division by zero
    scale = target_length / total_length
    scaled_transforms = [pin.SE3(transform.rotation, transform.translation * scale)
                         for transform in arm.link_transforms]
    return RobotArm(arm.nq, scaled_transforms, arm.joint_axes)


def add_intermediate_frames(arm: RobotArm, N=10, prefix=''):
    """
    Add intermediate frames along the links for visualization or IK purposes.
    """
    length = arm.length()
    s = np.linspace(0, length, N)[1:]  # Generate N points along the arm's length, ignoring the base
    link_lengths = np.array([np.linalg.norm(transform.translation) for transform in arm.link_transforms])
    joint2lengths = np.cumsum(link_lengths)
    root_ixs = np.searchsorted(joint2lengths, s) - 1  # Find which joint each point belongs to

    for i in range(len(s)):
        ix = root_ixs[i]
        parent_id = int(ix + 1)  # +1 to account for universe joint
        parent_transform = arm.link_transforms[ix + 1]
        link_size = link_lengths[ix + 1]  # Length of the current link

        if link_size < 1e-5:
            alpha = 0
        else:
            alpha = (s[i] - joint2lengths[ix]) / link_size

        # Compute the transform for this intermediate frame
        intermediate_transform = pin.SE3(
            np.eye(3),
            alpha * parent_transform.translation
        )

        # Add this intermediate frame to the model for IK
        intermediate_id = arm.model.addFrame(
            pin.Frame(name=f"{prefix}intermediate_frame_{i}",
                      parent_joint=parent_id,
                      parent_frame=0,
                      placement=intermediate_transform,
                      type=pin.FrameType.OP_FRAME)
        )
        # Add this intermediate frame to the geometry model for visualization
        sphere_radius = 0.03
        sphere = pin.GeometryObject(
            f"{prefix}intermediate_frame_{i}_sphere",
            parent_id,
            intermediate_id,
            intermediate_transform,
            pin.hppfcl.Sphere(sphere_radius)
        )
        sphere.meshColor = np.array([0.3, 0.3, 1.0, 1.0])
        arm.geom_model.addGeometryObject(sphere)


if __name__ == '__main__':
    ll1 = [0, .33, .33, .33]
    axes1 = [
        (0, 0, 1),  # Joint 1: Revolute around Z-axis
        (0, 1, 0),  # Joint 2: Revolute around Y-axis
        (1, 0, 0),  # Joint 3: Revolute around X-axis
        (0, 0, 1),  # Joint 4: Revolute around Z-axis
    ]

    import build_robot
    model, geom_model = build_robot.build_simple_arm(ll1, axes1, visualize=True)
    arm = RobotArm(model, geom_model, Nintermediate=11)
    arm.display_state(q=np.arange(arm.nq), show_frames=True)

    import time
    time.sleep(1)

    # print("Original arm length:", arm.length())
    # normalized_arm = normalize_arm_length(arm, target_length=1.0)
    # print("Normalized arm length:", normalized_arm.length())
