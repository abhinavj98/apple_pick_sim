# harvest_worlds_v3 coverage

Candidates: seeds 300-323 x 128 = 3072 from `fruiting_system_ranges_rl_harvest_real_g05_m1.json`, grasps i.i.d. in a 60 deg cone about world -y (equator, robot side), roll +-90 deg, world-frame grasp (D17i/j), 500/30 post-grasp hold; 2459 passed both screening builds; `harvest_worlds_v3.jsonl` is a seeded (0) random 2000 of those.

2459/3072 worlds accepted

| knob | candidate range | accepted range | reject rate low / mid / high tercile |
| --- | --- | --- | --- |
| primary.density | 330.2 .. 1228 | 330.2 .. 1228 | 21% / 19% / 20% |
| spur.length | 0.08921 .. 0.1058 | 0.08921 .. 0.1058 | 20% / 19% / 21% |
| spur.radius | 0.004263 .. 0.005817 | 0.004263 .. 0.005817 | 18% / 19% / 23% |
| spur.density | 3022 .. 8185 | 3022 .. 8185 | 19% / 21% / 20% |
| spur.youngs_modulus_pa | 5.613e+07 .. 1.068e+09 | 5.613e+07 .. 1.068e+09 | 19% / 19% / 21% |
| spur.flexural_modulus_pa | 2.107e+06 .. 4.517e+07 | 2.107e+06 .. 4.517e+07 | 17% / 21% / 22% |
| spur.damping_ratio | 0.2002 .. 0.7166 | 0.2002 .. 0.7166 | 20% / 19% / 21% |
| stem.length | 0.01374 .. 0.01477 | 0.01374 .. 0.01477 | 21% / 19% / 20% |
| stem.youngs_modulus_pa | 1.184e+08 .. 5.62e+09 | 1.184e+08 .. 5.62e+09 | 19% / 19% / 21% |
| stem.flexural_modulus_pa | 2.967e+07 .. 1.99e+08 | 2.967e+07 .. 1.99e+08 | 21% / 21% / 18% |
| stem.damping_ratio | 0.2553 .. 0.7596 | 0.2553 .. 0.7596 | 18% / 21% / 20% |
| stem.elevation_deg | -89.25 .. -30.62 | -89.25 .. -30.62 | 19% / 20% / 20% |
| apple_radius | 0.03004 .. 0.03355 | 0.03004 .. 0.03355 | 23% / 19% / 18% |
| apple_density | 1615 .. 2232 | 1615 .. 2232 | 19% / 22% / 19% |
| support_kp | 186.2 .. 301.5 | 186.2 .. 301.5 | 20% / 21% / 20% |
| support_roll_kp | 0.4841 .. 1.013 | 0.4841 .. 1.013 | 20% / 20% / 20% |
| support_zeta | 0.2513 .. 0.7818 | 0.2513 .. 0.7818 | 20% / 20% / 20% |
| arm_link_mass_scale | 0.9 .. 1.1 | 0.9 .. 1.1 | 20% / 22% / 18% |
| arm_link_inertia_scale | 0.9 .. 1.1 | 0.9 .. 1.1 | 20% / 19% / 20% |
| arm_ee_mass_scale | 0.95 .. 1.05 | 0.95 .. 1.05 | 19% / 21% / 20% |
| arm_ee_inertia_scale | 0.95 .. 1.05 | 0.95 .. 1.05 | 19% / 19% / 22% |
| weld_polar_deg | 30.85 .. 148.9 | 31.22 .. 145.7 | 10% / 17% / 32% |
| weld_elevation_deg | -59.15 .. 58.91 | -58.78 .. 55.71 | 10% / 17% / 32% |
| weld_azimuth_deg | -149.9 .. -30.47 | -149.9 .. -30.47 | 20% / 27% / 13% |
| weld_roll_deg | -90 .. 89.96 | -90 .. 89.96 | 17% / 27% / 15% |
