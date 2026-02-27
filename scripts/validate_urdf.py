"""
validate_urdf.py -- URDF static validation and gravity test

Validates a bipedal robot URDF file using PyBullet:
  1. Loads URDF and reports joint configuration (DOF, types, limits)
  2. Runs a free-fall gravity test (no control input)
  3. Outputs a structured calibration report

Usage:
    python scripts/validate_urdf.py [--urdf path/to/model.urdf]
    python scripts/validate_urdf.py --use-builtin   # Use PyBullet's built-in humanoid
"""

import pybullet as p
import pybullet_data
import time
import argparse
import os
import sys
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_and_validate_urdf(urdf_path: str, use_gui: bool = False):
    """
    Load a URDF and generate a validation report.

    Returns a dict with joint info, mass, and gravity test results.
    """
    # Connect to PyBullet
    physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity
    p.setGravity(0, 0, -9.81)

    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")

    # Load robot URDF
    start_pos = [0, 0, 1.0]  # Start 1m above ground
    start_orn = p.getQuaternionFromEuler([0, 0, 0])

    try:
        robot_id = p.loadURDF(
            urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn,
            useFixedBase=False,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load URDF: {e}")
        p.disconnect()
        return None

    # ---- 1. Joint configuration report ----
    num_joints = p.getNumJoints(robot_id)
    print("=" * 60)
    print(f"URDF Validation Report: {os.path.basename(urdf_path)}")
    print("=" * 60)
    print(f"\nTotal joints: {num_joints}")

    joint_types_map = {
        p.JOINT_REVOLUTE: "revolute",
        p.JOINT_PRISMATIC: "prismatic",
        p.JOINT_SPHERICAL: "spherical",
        p.JOINT_PLANAR: "planar",
        p.JOINT_FIXED: "fixed",
    }

    joints = []
    active_dof = 0

    print(f"\n{'#':<4} {'Name':<30} {'Type':<12} {'Lower':<10} {'Upper':<10} {'MaxForce':<10}")
    print("-" * 76)

    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        lower_limit = info[8]
        upper_limit = info[9]
        max_force = info[10]
        link_name = info[12].decode("utf-8")

        type_str = joint_types_map.get(joint_type, f"unknown({joint_type})")
        is_active = joint_type != p.JOINT_FIXED

        joint_data = {
            "index": i,
            "name": joint_name,
            "type": type_str,
            "lower": lower_limit,
            "upper": upper_limit,
            "max_force": max_force,
            "link": link_name,
            "active": is_active,
        }
        joints.append(joint_data)

        if is_active:
            active_dof += 1
            print(
                f"{i:<4} {joint_name:<30} {type_str:<12} "
                f"{lower_limit:<10.3f} {upper_limit:<10.3f} {max_force:<10.1f}"
            )

    print(f"\nActive DOF: {active_dof}")
    print(f"Fixed joints: {num_joints - active_dof}")

    # ---- 2. Mass report ----
    total_mass = 0.0
    print(f"\n--- Link Masses ---")
    # Base link
    base_dynamics = p.getDynamicsInfo(robot_id, -1)
    base_mass = base_dynamics[0]
    total_mass += base_mass
    print(f"  base_link: {base_mass:.3f} kg")

    for i in range(num_joints):
        dynamics = p.getDynamicsInfo(robot_id, i)
        mass = dynamics[0]
        total_mass += mass
        if mass > 0.01:  # Only print non-trivial masses
            link_name = joints[i]["link"]
            print(f"  {link_name}: {mass:.3f} kg")

    print(f"  TOTAL: {total_mass:.3f} kg")

    # ---- 3. Gravity free-fall test ----
    print(f"\n--- Gravity Test (free-fall, 2 seconds) ---")
    dt = 1.0 / 240.0
    steps = int(2.0 / dt)  # 2 seconds

    initial_pos, _ = p.getBasePositionAndOrientation(robot_id)
    print(f"  Initial position: z = {initial_pos[2]:.4f} m")

    for step in range(steps):
        p.stepSimulation()
        if use_gui:
            time.sleep(dt)

    final_pos, final_orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(final_orn)

    print(f"  Final position:   z = {final_pos[2]:.4f} m")
    print(f"  Vertical drop:    dz = {initial_pos[2] - final_pos[2]:.4f} m")
    print(f"  Final orientation (deg): roll={math.degrees(euler[0]):.1f}, "
          f"pitch={math.degrees(euler[1]):.1f}, yaw={math.degrees(euler[2]):.1f}")

    on_ground = final_pos[2] < 0.5
    print(f"  Gravity response: {'OK - fell to ground' if on_ground else 'WARNING - still floating'}")

    # ---- 4. Collision check ----
    contacts = p.getContactPoints(robot_id, plane_id)
    print(f"  Ground contacts: {len(contacts)} points")

    print("\n" + "=" * 60)
    print("Validation COMPLETE")
    print("=" * 60)

    report = {
        "urdf": urdf_path,
        "total_joints": num_joints,
        "active_dof": active_dof,
        "total_mass": total_mass,
        "joints": joints,
        "gravity_test": {
            "initial_z": initial_pos[2],
            "final_z": final_pos[2],
            "on_ground": on_ground,
            "contacts": len(contacts),
        },
    }

    p.disconnect()
    return report


def create_simple_biped_urdf(output_path: str):
    """
    Generate a simplified 6-DOF biped URDF for development.

    This is a placeholder model inspired by Unitree G1 proportions:
    - Torso (base_link)
    - Left/right legs: hip (revolute) -> knee (revolute) -> ankle (revolute)
    - Total: 6 active DOF

    To be replaced with real G1 URDF when available.
    """
    urdf_content = """<?xml version="1.0"?>
<robot name="simple_biped">

  <!-- ============ MATERIALS ============ -->
  <material name="blue">
    <color rgba="0.2 0.4 0.8 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.2 0.2 1.0"/>
  </material>

  <!-- ============ TORSO ============ -->
  <link name="base_link">
    <visual>
      <geometry><box size="0.2 0.3 0.4"/></geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry><box size="0.2 0.3 0.4"/></geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.08" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <!-- ============ LEFT LEG ============ -->
  <!-- Left Hip -->
  <link name="left_upper_leg">
    <visual>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
      <origin xyz="0 0 -0.15"/>
      <material name="grey"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
      <origin xyz="0 0 -0.15"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.012" ixy="0" ixz="0" iyy="0.012" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="left_hip" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_leg"/>
    <origin xyz="0 0.12 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="5"/>
  </joint>

  <!-- Left Knee -->
  <link name="left_lower_leg">
    <visual>
      <geometry><cylinder radius="0.035" length="0.3"/></geometry>
      <origin xyz="0 0 -0.15"/>
      <material name="grey"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.035" length="0.3"/></geometry>
      <origin xyz="0 0 -0.15"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="left_knee" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.6" effort="50" velocity="5"/>
  </joint>

  <!-- Left Ankle -->
  <link name="left_foot">
    <visual>
      <geometry><box size="0.15 0.08 0.03"/></geometry>
      <origin xyz="0.03 0 -0.015"/>
      <material name="red"/>
    </visual>
    <collision>
      <geometry><box size="0.15 0.08 0.03"/></geometry>
      <origin xyz="0.03 0 -0.015"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0.03 0 -0.015"/>
      <inertia ixx="0.0003" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="left_ankle" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.78" upper="0.78" effort="30" velocity="5"/>
  </joint>

  <!-- ============ RIGHT LEG ============ -->
  <!-- Right Hip -->
  <link name="right_upper_leg">
    <visual>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
      <origin xyz="0 0 -0.15"/>
      <material name="grey"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
      <origin xyz="0 0 -0.15"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.012" ixy="0" ixz="0" iyy="0.012" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="right_hip" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_leg"/>
    <origin xyz="0 -0.12 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="5"/>
  </joint>

  <!-- Right Knee -->
  <link name="right_lower_leg">
    <visual>
      <geometry><cylinder radius="0.035" length="0.3"/></geometry>
      <origin xyz="0 0 -0.15"/>
      <material name="grey"/>
    </visual>
    <collision>
      <geometry><cylinder radius="0.035" length="0.3"/></geometry>
      <origin xyz="0 0 -0.15"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="right_knee" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.6" effort="50" velocity="5"/>
  </joint>

  <!-- Right Ankle -->
  <link name="right_foot">
    <visual>
      <geometry><box size="0.15 0.08 0.03"/></geometry>
      <origin xyz="0.03 0 -0.015"/>
      <material name="red"/>
    </visual>
    <collision>
      <geometry><box size="0.15 0.08 0.03"/></geometry>
      <origin xyz="0.03 0 -0.015"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0.03 0 -0.015"/>
      <inertia ixx="0.0003" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="right_ankle" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.78" upper="0.78" effort="30" velocity="5"/>
  </joint>

</robot>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(urdf_content)
    print(f"[OK] Simple biped URDF generated: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="URDF Validation & Gravity Test")
    parser.add_argument("--urdf", type=str, default=None, help="Path to URDF file")
    parser.add_argument("--use-builtin", action="store_true", help="Use PyBullet built-in humanoid")
    parser.add_argument("--generate", action="store_true", help="Generate simple biped URDF first")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    args = parser.parse_args()

    if args.generate or (args.urdf is None and not args.use_builtin):
        # Generate and validate our simplified biped
        urdf_path = os.path.join(PROJECT_ROOT, "envs", "simple_biped.urdf")
        create_simple_biped_urdf(urdf_path)
        load_and_validate_urdf(urdf_path, use_gui=args.gui)
    elif args.use_builtin:
        import pybullet_data
        urdf_path = os.path.join(pybullet_data.getDataPath(), "humanoid", "humanoid.urdf")
        load_and_validate_urdf(urdf_path, use_gui=args.gui)
    else:
        load_and_validate_urdf(args.urdf, use_gui=args.gui)


if __name__ == "__main__":
    main()
