import math
import numpy as np
import warp as wp

import newton
import newton.examples

class AppleModelBuilder(newton.ModelBuilder):
    """Subclass of ModelBuilder to add custom rod functionality of having varying radius.""" 
    def add_rod(
        self,
        positions: list[wp.vec3],
        quaternions: list[wp.quat],
        radius: float | list[float] = 0.1,
        cfg: newton.ModelBuilder.ShapeConfig | None = None,
        stretch_stiffness: float | None = None,
        stretch_damping: float | None = None,
        bend_stiffness: float | None = None,
        bend_damping: float | None = None,
        closed: bool = False,
        key: str | None = None,
        wrap_in_articulation: bool = True,
    ) -> tuple[list[int], list[int]]:
        if cfg is None:
            cfg = self.default_shape_cfg

        # Stretch defaults: high stiffness to keep neighboring capsules tightly coupled
        stretch_stiffness = 1.0e9 if stretch_stiffness is None else stretch_stiffness
        stretch_damping = 0.0 if stretch_damping is None else stretch_damping
        # Bend defaults: 0.0 (users must explicitly set for bending resistance)
        bend_stiffness = 0.0 if bend_stiffness is None else bend_stiffness
        bend_damping = 0.0 if bend_damping is None else bend_damping

        # Input validation
        if stretch_stiffness < 0.0 or bend_stiffness < 0.0:
            raise ValueError("add_rod: stretch_stiffness and bend_stiffness must be >= 0")

        # Guard against near-zero lengths: segment length is used to normalize stiffness later (EA/L, EI/L).
        min_segment_length = 1.0e-9
        num_segments = len(quaternions)
        if len(positions) != num_segments + 1:
            raise ValueError(
                f"add_rod: positions must have {num_segments + 1} elements for {num_segments} segments, "
                f"got {len(positions)} positions"
            )
        if num_segments < 2:
            # A "rod" in this API is defined as multiple capsules coupled by cable joints.
            # If you want a single capsule, create a body + capsule shape directly.
            raise ValueError(
                f"add_rod: requires at least 2 segments (got {num_segments}); "
                "for a single capsule, create a body and add a capsule shape instead."
            )

        link_bodies = []
        link_joints = []
        segment_lengths: list[float] = []

        # Create all bodies first
        for i in range(num_segments):
            p0 = positions[i]
            p1 = positions[i + 1]
            q = quaternions[i]

            # Calculate segment properties
            segment_length = wp.length(p1 - p0)
            if segment_length <= min_segment_length:
                raise ValueError(
                    f"add_rod: segment {i} has a too-small length (length={float(segment_length):.3e}); "
                    f"segment length must be > {min_segment_length:.1e}"
                )
            segment_lengths.append(float(segment_length))
            half_height = 0.5 * segment_length

            # Sanity check: ensure the capsule orientation aligns its local +Z axis with
            # the segment direction between positions[i] and positions[i+1]. This enforces
            # the contract that ``quaternions[i]`` is a world-space rotation taking local +Z
            # into ``positions[i+1] - positions[i]``; otherwise the capsules will not form
            # a proper rod.
            seg_dir = wp.normalize(p1 - p0)
            local_z_world = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
            alignment = wp.dot(seg_dir, local_z_world)
            if alignment < 0.999:
                raise ValueError(
                    "add_rod: quaternion at index "
                    f"{i} does not align capsule +Z with segment (positions[i+1] - positions[i]); "
                    "quaternions must be world-space and constructed so that local +Z maps to the "
                    "segment direction positions[i+1] - positions[i]."
                )

            # Position body at start point, with COM offset to segment center
            body_q = wp.transform(p0, q)

            # COM offset in local coordinates: from start point to center
            com_offset = wp.vec3(0.0, 0.0, half_height)

            # Generate unique keys for each entity type to avoid conflicts
            body_key = f"{key}_body_{i}" if key else None
            shape_key = f"{key}_capsule_{i}" if key else None

            child_body = self.add_link(xform=body_q, com=com_offset, key=body_key)

            # Place capsule so it spans from start to end along +Z
            capsule_xform = wp.transform(wp.vec3(0.0, 0.0, half_height), wp.quat_identity())
            
            # Determine radius for this segment
            if isinstance(radius, (int, float)):
                r = float(radius)
            elif len(radius) == num_segments:
                r = radius[i]
            elif len(radius) == num_segments + 1:
                # Interpolate radius from endpoints
                r = 0.5 * (radius[i] + radius[i + 1])
            else:
                raise ValueError(
                    f"add_rod: radius must be a float or a sequence of length {num_segments} or {num_segments + 1} "
                    f"(got sequence of length {len(radius)})"
                )

            self.add_shape_capsule(
                child_body,
                xform=capsule_xform,
                radius=r,
                half_height=half_height,
                cfg=cfg,
                key=shape_key,
            )
            link_bodies.append(child_body)

        # Create joints connecting consecutive segments
        # For open chains: num_segments - 1 joints
        # For closed loops: num_segments joints (including closing joint)
        num_joints = num_segments if closed else num_segments - 1
        for i in range(num_joints):
            parent_idx = i
            child_idx = (i + 1) % num_segments  # Wraps around for closing joint when closed

            parent_body = link_bodies[parent_idx]
            child_body = link_bodies[child_idx]
            if parent_body == child_body:
                raise ValueError(
                    "add_rod: invalid rod topology; attempted to create a joint connecting a body to itself. "
                    "This should be unreachable (add_rod requires >=2 segments)."
                )

            # Parent anchor at segment end
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, segment_lengths[parent_idx]), wp.quat_identity())

            # Child anchor at segment start
            child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

            # Joint key: numbered 1 through num_joints
            joint_key = f"{key}_cable_{i + 1}" if key else None

            # Pre-scale rod stiffnesses here so solver kernels do not need per-segment length normalization.
            # Use the parent segment length L.
            #
            # - Stretch: treat the user input as a material-like axial/shear stiffness (commonly EA) [N]
            #   and store an effective per-joint (point-to-point) spring stiffness k_eff = EA / L [N/m].
            # - Bend/twist: treat the user input as a material-like bending/twist stiffness (commonly EI) [N*m^2]
            #   and store an effective per-joint angular stiffness k_eff = EI / L [N*m].
            #
            # TODO: For physically accurate tapered rods, the stiffness properties (EA and EI) should technically
            # scale with the varying radius (EA ~ r^2, EI ~ r^4). Currently, we use uniform material properties
            # which implies the thinner segments are made of a "stiffer" material to compensate.
            seg_len = segment_lengths[parent_idx]
            stretch_ke_eff = stretch_stiffness / seg_len
            bend_ke_eff = bend_stiffness / seg_len

            joint = self.add_joint_cable(
                parent=parent_body,
                child=child_body,
                parent_xform=parent_xform,
                child_xform=child_xform,
                bend_stiffness=bend_ke_eff,
                bend_damping=bend_damping,
                stretch_stiffness=stretch_ke_eff,
                stretch_damping=stretch_damping,
                key=joint_key,
                collision_filter_parent=True,
                enabled=True,
            )
            link_joints.append(joint)

        # Optionally (by default) wrap all rod joints into a single articulation.
        if wrap_in_articulation and link_joints:
            # Derive a default articulation key if none is provided.
            rod_art_key = f"{key}_articulation" if key else None

            self.add_articulation(link_joints, key=rod_art_key)

        return link_bodies, link_joints

class ExampleAppleStem:
    def create_stem_geometry(
        self,
        pos: wp.vec3,
        num_elements: int,
        length: float,
    ):
        """Straight stiff stem"""
        num_points = num_elements + 1
        points = []

        # Create points along the stem
        for i in range(num_points):
            t = i / num_elements
            points.append(pos + wp.vec3(length * t, 0.0, 0.0))

        edge_q = []
        local_axis = wp.vec3(0.0, 0.0, 1.0) #Z axis is the local axis
        from_dir = local_axis

        # Create quaternions for parallel transport
        for i in range(num_elements):
            p0, p1 = points[i], points[i + 1]
            to_dir = wp.normalize(p1 - p0)

            dq = wp.quat_between_vectors(from_dir, to_dir)
            if i == 0:
                q = dq
            else:
                q = wp.mul(dq, edge_q[i - 1])

            edge_q.append(q)
            from_dir = to_dir

        return points, edge_q

    def __init__(self, viewer, args=None):
        # ------------------------------------------------------------
        # Simulation params
        # ------------------------------------------------------------
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.sim_iterations = 100
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ------------------------------------------------------------
        # Stem parameters
        # ------------------------------------------------------------
        self.num_elements = 5
        self.segment_length = 0.1
        self.stem_length = self.num_elements * self.segment_length
        self.stem_radius = 0.01
        self.stem_density = 200

        # Each entry defines a lateral offset and bend stiffness so we can compare behaviors side-by-side.
        lateral_spacing = 0.35
        self.stem_variants = [
            {"label": "soft", "y_offset": -lateral_spacing, "bend_stiffness": 1.0e0, "bend_damping": 2.0e-1},
            {"label": "medium", "y_offset": 0.0, "bend_stiffness": 1.0e2, "bend_damping": 2.0e-1},
            {"label": "stiff", "y_offset": lateral_spacing, "bend_stiffness": 1.0e4, "bend_damping": 2.0e-1},
        ]
        self.stem_stretch_stiffness = 1.0e9
        
        # ------------------------------------------------------------
        # Apple parameters
        # ------------------------------------------------------------
        apple_radius = 0.06
        apple_density = 850.0  # kg/m^3 (fruit-like)

        apple_mass = (4.0 / 3.0) * np.pi * apple_radius**3 * apple_density
        stem_mass = self.num_elements * self.segment_length * np.pi * self.stem_radius**2 * self.stem_density
        self.break_force = 50 + (apple_mass+stem_mass)*9.81 #Break force + weight of apple and stem
        self.break_torque = 20 #Nm
       

        # ------------------------------------------------------------
        # Build model
        # ------------------------------------------------------------
        builder = AppleModelBuilder()

        builder.default_shape_cfg.ke = 1.0e2
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 0.8

        self.stem_tip_bodies: list[int] = []
        self.apple_bodies: list[int] = []
        self.apple_joints: list[int] = []
        self.broken_joints: set[int] = set()
        
        # Build a row of stems with varying stiffness so the apples react differently.
        for variant in self.stem_variants:
            start_pos = wp.vec3(-self.stem_length * 0.5, variant["y_offset"], 3.0)
            stem_points, stem_edge_q = self.create_stem_geometry(
                start_pos,
                self.num_elements,
                self.stem_length,
            )
            #Last segment is thinner and smaller
            last_segment_dir = wp.normalize(stem_points[-1] - stem_points[-2])
            stem_points[-1] = stem_points[-2] + last_segment_dir * (self.stem_length / 10.0)
            radius = self.stem_radius * np.ones(self.num_elements + 1)
            radius[-1] = self.stem_radius/10


            stem_cfg = builder.ShapeConfig(
                density=self.stem_density
            )

            stem_bodies, stem_joints = builder.add_rod(
                positions=stem_points,
                quaternions=stem_edge_q,
                radius=radius,
                bend_stiffness=variant["bend_stiffness"],
                bend_damping=variant["bend_damping"],
                stretch_stiffness=self.stem_stretch_stiffness,
                stretch_damping=0.0,
                cfg=stem_cfg,
                wrap_in_articulation=False,
                key=f"stem_{variant['label']}",
            )

            base_body = stem_bodies[0]
            builder.body_mass[base_body] = 0.0
            builder.body_inv_mass[base_body] = 0.0

            # --------------------------------------------------------
            # Apple rigid body
            # --------------------------------------------------------
            stem_tip = stem_points[-1]
            stem_prev = stem_points[-2]
            stem_dir = wp.normalize(stem_tip - stem_prev)
            attach_offset = 0.0
            apple_pos = stem_tip + stem_dir * attach_offset

            apple_body = builder.add_link(
                xform=wp.transform(apple_pos, wp.quat_identity()),
                mass=apple_mass,
            )

            builder.add_shape_sphere(
                body=apple_body,
                radius=apple_radius,
            )

            # Attach apple
            stem_tip_body = stem_bodies[-1]
            stem_tip_quat = stem_edge_q[-1]
            segment_vector_world = stem_dir * self.segment_length
            parent_local_anchor = wp.quat_rotate(wp.quat_inverse(stem_tip_quat), segment_vector_world)
            child_local_anchor = -stem_dir * attach_offset
            apple_joint = builder.add_joint_fixed(
                parent=stem_tip_body,
                child=apple_body,
                parent_xform=wp.transform(parent_local_anchor, wp.quat_identity()),
                child_xform=wp.transform(child_local_anchor, wp.quat_identity()),
            ) #Only ball is supported by VBD solver

            builder.add_articulation([*stem_joints, apple_joint])

            self.stem_tip_bodies.append(stem_tip_body)
            self.apple_bodies.append(apple_body)
            self.apple_joints.append(apple_joint)

        # Ground
        builder.add_ground_plane()

        # Coloring
        builder.color()

        # ------------------------------------------------------------
        # Finalize
        # ------------------------------------------------------------
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=0.1,
            rigid_contact_k_start=1.0e4,
            rigid_joint_linear_k_start=1.0e8,
            rigid_joint_angular_k_start=1.0e6,
        )

        # Cache solver constraint starts for breakable joint logic
        self.solver_constraint_starts = self.solver.joint_constraint_start.to("cpu").numpy()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.rest_joint_distances = self._measure_rest_joint_distances()

        # Reference distance between apple body and stem tip body in rest pose
        # rest_body_q = self.state_0.body_q.numpy()
        # self.rest_joint_distance = np.linalg.norm(
        #     rest_body_q[self.apple_body, :3] - rest_body_q[self.stem_tip_body, :3]
        # )

        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args
        )
        self.contacts = self.model.collide(
            self.state_0, collision_pipeline=self.collision_pipeline
        )

        self.viewer.set_model(self.model)

        self.capture()

    def get_forces(self):
        """Logs forces on the last stem segment (thinner part) and the apple joint."""

        #TODO: Forces are calculated via distance constraints in the solver. 
        # The force is calculated as F = k * (d - d_rest)
        # where k is the penalty stiffness and d_rest is the rest length.
        # TODO: Do we need a damping term? 


        # Only log forces periodically
        # if self.sim_time % 0.1 < self.sim_dt:
        torques = []
        linear_forces = []
        print(f"--- Forces at t={self.sim_time:.3f} ---")
        
        # Access solver arrays
        joint_penalty_k = self.solver.joint_penalty_k.to("cpu").numpy()
        constraint_starts = self.solver_constraint_starts
        
        # Helper to get body transforms
        body_q = self.state_0.body_q.to("cpu").numpy()
        def get_xform(body_idx):
            p = body_q[body_idx]
            return wp.transform(wp.vec3(float(p[0]), float(p[1]), float(p[2])), wp.quat(float(p[3]), float(p[4]), float(p[5]), float(p[6])))

        # Helper to get joint anchor transforms
        joint_X_p = self.model.joint_X_p.to("cpu").numpy()
        joint_X_c = self.model.joint_X_c.to("cpu").numpy()
        def get_anchor_xforms(joint_idx):
            jp = joint_X_p[joint_idx]
            X_bp_j = wp.transform(wp.vec3(float(jp[0]), float(jp[1]), float(jp[2])), wp.quat(float(jp[3]), float(p[4]), float(jp[5]), float(jp[6])))
            jc = joint_X_c[joint_idx]
            X_bc_j = wp.transform(wp.vec3(float(jc[0]), float(jc[1]), float(jc[2])), wp.quat(float(jc[3]), float(jc[4]), float(jc[5]), float(jc[6])))
            return X_bp_j, X_bc_j

        # Iterate over each stem variant
        for i, variant in enumerate(self.stem_variants):
            if i >= len(self.apple_joints): continue

            # 1. Apple Joint (BALL joint)
            apple_joint = self.apple_joints[i]
            
            # 2. Last Stem Joint (CABLE joint)
            # Assuming last_stem_joint = apple_joint - 1
            last_stem_joint = apple_joint - 1
            
            c0 = int(constraint_starts[last_stem_joint])
            k_stretch = joint_penalty_k[c0]
            k_bend = joint_penalty_k[c0 + 1]
            
            parent_body = int(self.model.joint_parent.numpy()[last_stem_joint])
            child_body = int(self.model.joint_child.numpy()[last_stem_joint])
            
            # Check if this looks like a valid stem joint (not -1)
            if parent_body >= 0:
                X_wb_p = get_xform(parent_body)
                X_wb_c = get_xform(child_body)
                
                jp = joint_X_p[last_stem_joint]
                X_bp_j = wp.transform(wp.vec3(float(jp[0]), float(jp[1]), float(jp[2])), wp.quat(float(jp[3]), float(jp[4]), float(jp[5]), float(jp[6])))
                jc = joint_X_c[last_stem_joint]
                X_bc_j = wp.transform(wp.vec3(float(jc[0]), float(jc[1]), float(jc[2])), wp.quat(float(jc[3]), float(jc[4]), float(jc[5]), float(jc[6])))
                
                pos_anchor_p = wp.transform_point(X_wb_p, wp.transform_get_translation(X_bp_j))
                pos_anchor_c = wp.transform_point(X_wb_c, wp.transform_get_translation(X_bc_j))
                
                dist_lin = wp.length(pos_anchor_p - pos_anchor_c)
                force_lin = k_stretch * dist_lin
                
                # Angular deviation
                rot_p = wp.mul(wp.transform_get_rotation(X_wb_p), wp.transform_get_rotation(X_bp_j))
                rot_c = wp.mul(wp.transform_get_rotation(X_wb_c), wp.transform_get_rotation(X_bc_j))
                
                z_axis = wp.vec3(0.0, 0.0, 1.0)
                z_p = wp.quat_rotate(rot_p, z_axis)
                z_c = wp.quat_rotate(rot_c, z_axis)
                
                dot_val = wp.dot(z_p, z_c)
                # clamp to avoid nan
                if dot_val > 1.0: dot_val = 1.0
                elif dot_val < -1.0: dot_val = -1.0
                angle = math.acos(dot_val)
                
                torque_bend = k_bend * angle

                torques.append(torque_bend)
                linear_forces.append(force_lin)
                
                print(f"Last Stem Joint (cable): Force={force_lin:.4f} N (distance={dist_lin:.6f} m), Torque={torque_bend:.4f} Nm (angle={math.degrees(angle):.2f} deg)")

        return torques, linear_forces

    def capture(self):
        if self.solver.device.is_cuda:
            self.capturing = True
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.capturing = False
            self.graph = cap.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(
                self.state_0, collision_pipeline=self.collision_pipeline
            )

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0


    def step(self, warmup = False):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        #TODO: Make this get forces calculation as a kernel
        if not warmup:
            torques, linear_forces = self.get_forces()
            #TODO: Need a low pass filter for the forces to avoid false positives
            #Check if any forces exceed the break threshold and print joint breakage
            for i, (torque, linear_force) in enumerate(zip(torques, linear_forces)):
                if torque > self.break_torque or linear_force > self.break_force:
                    print(f"Joint {i} broken!")   
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_apple_attachment(self, tolerance: float = 1e-2, warmup_steps: int = 5):
        """Checks that the fixed joint keeps the apple locked to the stem tip."""

        for _ in range(warmup_steps):
            self.step()

        body_q = self.state_0.body_q.numpy()
        for rest_distance, stem_body, apple_body in zip(
            self.rest_joint_distances, self.stem_tip_bodies, self.apple_bodies
        ):
            current_distance = np.linalg.norm(
                body_q[apple_body, :3] - body_q[stem_body, :3]
            )
            assert abs(current_distance - rest_distance) < tolerance, "Apple drifted too far from stem tip"

    def _measure_rest_joint_distances(self):
        body_q = self.state_0.body_q.numpy()
        distances = []
        for stem_body, apple_body in zip(self.stem_tip_bodies, self.apple_bodies):
            distances.append(
                np.linalg.norm(body_q[apple_body, :3] - body_q[stem_body, :3])
            )
        return distances


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = ExampleAppleStem(viewer, args)
    #Warmup to let the system settle
    print("Warming up...")
    for _ in range(100):
        example.step(warmup = True)

    print("Starting simulation...")
    while True:
        example.step()
        example.render()
