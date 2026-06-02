import warp as wp
import numpy as np
wp.init()

off = wp.vec3(0, 0, -0.1)
final_rot = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi/2)

# proxy_offset_in_apple_frame
go = (off[0], off[1], off[2], final_rot[0], final_rot[1], final_rot[2], final_rot[3])

offset_tf = wp.transform(
    wp.vec3(float(go[0]), float(go[1]), float(go[2])),
    wp.quat(float(go[3]), float(go[4]), float(go[5]), float(go[6])),
)

tcp_pos = wp.vec3(1, 1, 1)
tcp_rot = wp.quat_identity()
tcp_tf = wp.transform(tcp_pos, tcp_rot)

# New way
apple_tf = wp.transform_multiply(tcp_tf, wp.transform_inverse(offset_tf))
print("New p_apple:", wp.transform_get_translation(apple_tf))

# Old way
delta = wp.quat_rotate(tcp_rot, wp.vec3(off[0], off[1], off[2]))
print("Old p_apple:", tcp_pos - delta)
