import warp as wp
wp.init()
q1 = wp.quat_identity()
q2 = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.0)
print(type(q1))
try:
    print("mul:", wp.mul(q1, q2))
except Exception as e:
    print("mul err:", e)
try:
    print("*:", q1 * q2)
except Exception as e:
    print("* err:", e)
