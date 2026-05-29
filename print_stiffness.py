import os
import sys
import numpy as np

# Ensure both workspace root and newton directories are on the path
# If run with --directory newton, the working directory is /home/abhinav/codes/apple_pick_sim/newton
cwd = os.getcwd()
if cwd.endswith("newton"):
    repo_root = os.path.dirname(cwd)
else:
    repo_root = cwd

sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "newton"))

import warp as wp
wp.init()

import apple_pick_sim.coupled_fruiting as cf
import apple_pick_sim.fruiting_system as fs

# 1. Load ranges
ranges_path = os.path.join(repo_root, "apple_pick_sim", "fixtures", "fruiting_system_ranges_example_variance.json")
ranges = fs.load_ranges(ranges_path)

# 2. Build the mega scene (same config as your keyboard teleop command)
gripper = fs.GripperProxyConfig(mass=2.0, box_half_extents=(0.05, 0.05, 0.05), fix_to_apple=True)
scene = cf.build_mega_coupled_fruiting_fr3(
    ranges,
    seed=0,
    gripper_proxy=gripper,
    stiffness_epsilon=0.2,
    enable_self_collisions=False,
    mujoco_solver_kwargs={"disable_contacts": True},
)

mega = scene.cable
solver = mega.solver
model = mega.model

jc_start = solver.joint_constraint_start.numpy()
joint_penalty_k = solver.joint_penalty_k.numpy()
jchild = model.joint_child.numpy()

from apple_pick_sim.fruiting_system.mega_fd import reset_perturbed_instances_to_nominal

def print_stiffnesses(label: str) -> None:
    print(f"\nStiffness Values [ {label} ]:")
    print("=" * 85)
    # Read fresh penalty array from device
    joint_penalty_k = solver.joint_penalty_k.numpy()
    for col in range(mega.num_instances):
        inst = mega.instance(col)
        print(f"Instance {col} (offset base: {inst.base_pos}):")
        
        segments = {
            "primary": inst.primary_bodies,
            "secondary": inst.secondary_bodies,
            "spur": inst.spur_bodies,
            "stem": inst.stem_bodies
        }
        
        for seg_name, bodies in segments.items():
            if not bodies:
                continue
            seg_joint = None
            for j_idx in inst.cable_joint_indices:
                child = int(jchild[j_idx])
                if child in bodies:
                    seg_joint = j_idx
                    break
                    
            if seg_joint is not None:
                c0 = int(jc_start[seg_joint])
                bend_k = joint_penalty_k[c0 + 1]
                print(f"  First {seg_name:9s} joint (ID {seg_joint:3d}): bend_stiffness = {bend_k:.4f}")
        print("-" * 85)

print_stiffnesses("BEFORE reset_perturbed_instances_to_nominal")

print("\nCalling reset_perturbed_instances_to_nominal(scene.cable, nominal_index=0)...")
reset_perturbed_instances_to_nominal(mega, nominal_index=scene.nominal_index)

print_stiffnesses("AFTER reset_perturbed_instances_to_nominal")

